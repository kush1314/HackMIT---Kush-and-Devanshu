from flask import Flask, request, render_template_string, send_file
from PIL import Image
import numpy as np
import io
import hashlib
import time

app = Flask(__name__)

# In-memory database
WATERMARK_DB = {}  # {hash: info}


SECRET_KEY = "HackMIT2025Secret"  # For deterministic hashing testing


def generate_hash(prompt: str) -> str:
    # generates a unique hash for an image using prompt + timestamp + secret key
    raw = f"{prompt}-{time.time()}-{SECRET_KEY}".encode()
    return hashlib.sha256(raw).hexdigest()[:64] 


def encode_watermark(image: Image.Image, watermark: str) -> Image.Image:
    #eEmbed watermark invisibly on RGB values, making sure to preserve alpha (_, _, _, x)
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    h, w, _ = arr.shape

    # Convert watermark hex -> binary string
    bin_str = ''.join(format(int(c, 16), '04b') for c in watermark)
    total_pixels = h * w * 3  # rgb
    if len(bin_str) > total_pixels:
        raise ValueError("Image too small for watermark!")
    flat = arr[:, :, :3].flatten()

    # Encode bits
    for i, bit in enumerate(bin_str):
        flat[i] = np.uint8((flat[i] & 0b11111110) | int(bit))

    # Put back
    arr[:, :, :3] = flat.reshape((h, w, 3))
    return Image.fromarray(arr, 'RGBA')


def decode_watermark(image: Image.Image, length=64) -> str:
    # Extract watermark (hex string) from RGB LSBs
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    flat = arr[:, :, :3].flatten()

    bin_str = ''.join(str(int(flat[i] & 1)) for i in range(length * 4))
    hex_str = ''.join(format(int(bin_str[i:i+4], 2), 'x') for i in range(0, len(bin_str), 4))
    return hex_str[:length]


# --- Flask Routes ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>AI Watermark Demo</title></head>
<body>
<h2>Upload PNG to Encode Watermark</h2>
<form method="POST" action="/encode" enctype="multipart/form-data">
  Information: <input type="text" name="prompt" required><br><br>
  <input type="file" name="file" accept="image/png" required>
  <input type="submit" value="Encode & Download">
</form>

<hr>
<h2>Upload PNG to Verify Watermark</h2>
<form method="POST" action="/verify" enctype="multipart/form-data">
  <input type="file" name="file" accept="image/png" required>
  <input type="submit" value="Verify">
</form>

{% if result %}
  <h3>Verification Result:</h3>
  <p>{{ result }}</p>
{% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


@app.route("/encode", methods=["POST"])
def encode():
    prompt = request.form["prompt"]
    file = request.files["file"]
    image = Image.open(file.stream)

    wm_hash = generate_hash(prompt)

    stamped = encode_watermark(image, wm_hash)

    # Store in DB for verification later
    WATERMARK_DB[wm_hash] = {"prompt": prompt, "timestamp": time.time()}

    # Return encoded image for download
    buf = io.BytesIO()
    stamped.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", as_attachment=True, download_name="encoded.png")


@app.route("/verify", methods=["POST"])
def verify():
    file = request.files["file"]
    image = Image.open(file.stream)

    try:
        wm_hash = decode_watermark(image)
    except Exception:
        return render_template_string(HTML_PAGE, result="❌ Failed to decode watermark.")

    # SHA-256 is found in SQLite
    if wm_hash in WATERMARK_DB:
        info = WATERMARK_DB[wm_hash]
        result = f"This image has been hashed, it is AI-generated! (info: '{info['prompt']}', hash: {wm_hash})"
    else:
        result = "No watermark detected (likely not AI-generated)"

    return render_template_string(HTML_PAGE, result=result)


if __name__ == "__main__":
    app.run(debug=True)
