from flask import Flask, request, render_template, send_file
from PIL import Image
import numpy as np
import io
import hashlib
import sqlite3
import os
from dotenv import load_dotenv
from PIL import ImageDraw

app = Flask(__name__)
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "hackmitsecret")
DB_PATH = "watermarks.db"

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS watermarks (
            hash TEXT PRIMARY KEY,
            prompt TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- Watermark Functions ---
def generate_hash(prompt: str) -> str:
    return hashlib.sha256((prompt + SECRET_KEY).encode()).hexdigest()[:64]

# def encode_watermark(image: Image.Image, wm_hash: str) -> Image.Image:
#     arr = np.array(image.convert("RGBA"), dtype=np.uint8)
#     h, w, _ = arr.shape

#     # Convert hex hash to binary string
#     bin_str = ''.join(format(int(c, 16), '04b') for c in wm_hash)
#     total_pixels = h * w * 3
#     if len(bin_str) > total_pixels:
#         raise ValueError("Image too small for watermark!")

#     flat = arr[:, :, :3].flatten()
#     for i, bit in enumerate(bin_str):
#         flat[i] = np.uint8((flat[i] & 0b11111110) | int(bit))
#     arr[:, :, :3] = flat.reshape((h, w, 3))
#     return Image.fromarray(arr, 'RGBA')

def encode_watermark(image: Image.Image, wm_hash: str) -> Image.Image:
    """
    Encode watermark bits into the LSBs of the RGB channels, distributed evenly across the image.
    """
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    h, w, _ = arr.shape
    flat = arr[:, :, :3].flatten()

    # Convert hex hash to binary string
    bin_str = ''.join(format(int(c, 16), '04b') for c in wm_hash)
    bin_len = len(bin_str)

    if bin_len > len(flat):
        raise ValueError("Image too small for watermark!")

    # Spread watermark bits across the flat array
    step = len(flat) // bin_len
    for i, bit in enumerate(bin_str):
        idx = i * step
        flat[idx] = np.uint8((flat[idx] & 0b11111110) | int(bit))

    arr[:, :, :3] = flat.reshape((h, w, 3))
    return Image.fromarray(arr, 'RGBA')


# def decode_watermark(image: Image.Image, length=64) -> str:
#     arr = np.array(image.convert("RGBA"), dtype=np.uint8)
#     flat = arr[:, :, :3].flatten()
#     bin_str = ''.join(str(flat[i] & 1) for i in range(length * 4))
#     hex_str = ''.join(format(int(bin_str[i:i+4], 2), 'x') for i in range(0, len(bin_str), 4))
#     return hex_str[:length]

def decode_watermark(image: Image.Image, length=64) -> str:
    """
    Decode watermark bits from an image where bits are evenly distributed.
    """
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    flat = arr[:, :, :3].flatten()

    total_bits = length * 4
    step = len(flat) // total_bits

    bin_str = ''
    for i in range(total_bits):
        idx = i * step
        bin_str += str(flat[idx] & 1)

    # Convert back to hex
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

from PIL import ImageDraw

def highlight_watermark_pixels(image: Image.Image, length=64) -> Image.Image:
    """
    Highlight watermark pixels across the whole image, matching the evenly-distributed encoding.
    """
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    h, w, _ = arr.shape
    flat = arr[:, :, :3].flatten()

    total_bits = length * 4
    step = len(flat) // total_bits

    marked_pixels = []
    for i in range(total_bits):
        idx = i * step
        if flat[idx] & 1:
            pixel_idx = idx // 3
            y = pixel_idx // w
            x = pixel_idx % w
            marked_pixels.append((x, y))

    highlighted_image = image.convert("RGBA")
    draw = ImageDraw.Draw(highlighted_image)
    square_size = 8  # bigger squares for visibility

    for x, y in marked_pixels:
        top_left = (max(x - square_size//2, 0), max(y - square_size//2, 0))
        bottom_right = (min(x + square_size//2, w-1), min(y + square_size//2, h-1))
        draw.rectangle([top_left, bottom_right], fill=(255,0,0,180))  # semi-transparent red

    return highlighted_image




# --- Flask Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

@app.route("/encode", methods=["POST"])
def encode():
    prompt = request.form["prompt"]
    file = request.files["file"]
    image = Image.open(file.stream)

    try:
        wm_hash = generate_hash(prompt)
        stamped = encode_watermark(image, wm_hash)
        save_watermark(wm_hash, prompt)

        buf = io.BytesIO()
        stamped.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="encoded.png")
    except Exception as e:
        return render_template("index.html", result=f"❌ Failed to encode: {str(e)}")

@app.route("/verify", methods=["POST"])
@app.route("/verify", methods=["POST"])
def verify():
    file = request.files["file"]
    image = Image.open(file.stream)

    highlighted_image = None
    try:
        extracted_hash = decode_watermark(image)
        prompt = get_prompt_by_hash(extracted_hash)
        if prompt:
            result = f"✅ Watermark detected! Original prompt: '{prompt}'"
            # Generate highlighted image
            highlighted_image = highlight_watermark_pixels(image)
            # Convert to base64 for HTML embedding
            buf = io.BytesIO()
            highlighted_image.save(buf, format="PNG")
            buf.seek(0)
            import base64
            img_data = base64.b64encode(buf.getvalue()).decode()
            highlighted_image = f"data:image/png;base64,{img_data}"
        else:
            result = "❌ No watermark detected."
    except Exception as e:
        result = f"❌ Failed to decode watermark: {str(e)}"

    return render_template("index.html", result=result, highlighted_image=highlighted_image)


if __name__ == "__main__":
    app.run(debug=True)
