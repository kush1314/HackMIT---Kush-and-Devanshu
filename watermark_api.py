from flask import Flask, request, render_template, send_file
from PIL import Image
import numpy as np
import io
import hashlib
import sqlite3
import os
from dotenv import load_dotenv


app = Flask(__name__)

load_dotenv() 
SECRET_KEY = os.getenv("SECRET_KEY")
DB_PATH = "watermarks.db"


def generate_hash(prompt: str) -> str:
    # deterministic hash based on prompt and SECRET_KEY 
    return hashlib.sha256((prompt + SECRET_KEY).encode()).hexdigest()[:64]

def encode_watermark(image: Image.Image, wm_hash: str) -> Image.Image:
    # embed watermark invisibly LSBs
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    h, w, _ = arr.shape

    bin_str = ''.join(format(int(c, 16), '04b') for c in wm_hash)
    total_pixels = h * w * 3
    if len(bin_str) > total_pixels:
        raise ValueError("Image too small for watermark!")

    flat = arr[:, :, :3].flatten()
    for i, bit in enumerate(bin_str):
        flat[i] = np.uint8((flat[i] & 0b11111110) | int(bit))
    arr[:, :, :3] = flat.reshape((h, w, 3))
    return Image.fromarray(arr, 'RGBA')

def decode_watermark(image: Image.Image, length=64) -> str:
    # extract watermark from RGB LSBs
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    flat = arr[:, :, :3].flatten()
    bin_str = ''.join(str(flat[i] & 1) for i in range(length * 4))
    hex_str = ''.join(format(int(bin_str[i:i+4], 2), 'x') for i in range(0, len(bin_str), 4))
    return hex_str[:length]

def save_watermark(wm_hash, prompt):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO watermarks (hash, prompt) VALUES (?, ?)", (wm_hash, prompt))
    conn.commit()
    conn.close()

def get_prompt_by_hash(wm_hash):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT prompt FROM watermarks WHERE hash=?", (wm_hash,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None



# --- Flask Stuff ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

@app.route("/encode", methods=["POST"])
def encode():
    prompt = request.form["prompt"]
    file = request.files["file"]
    image = Image.open(file.stream)

    wm_hash = generate_hash(prompt)
    stamped = encode_watermark(image, wm_hash)

    # Save hash + prompt in DB
    save_watermark(wm_hash, prompt)

    buf = io.BytesIO()
    stamped.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", as_attachment=True, download_name="encoded.png")

@app.route("/verify", methods=["POST"])
def verify():
    file = request.files["file"]
    image = Image.open(file.stream)

    try:
        extracted_hash = decode_watermark(image)
        prompt = get_prompt_by_hash(extracted_hash)
        if prompt:
            result = f"✅ Watermark detected! Original prompt: '{prompt}'"
        else:
            result = "❌ No watermark detected."
    except Exception:
        result = "❌ Failed to decode watermark."

    return render_template("index.html", result=result)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
