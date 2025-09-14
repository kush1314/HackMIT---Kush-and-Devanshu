from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image
import numpy as np
import io
import hashlib
import sqlite3
import os
from dotenv import load_dotenv
from PIL import ImageDraw
import requests
import time
import re
import base64
import anthropic
import random

app = Flask(__name__)
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_PATH = "watermarks.db"

# Initialize Anthropic client
if ANTHROPIC_API_KEY:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    client = None

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

# --- Image Generation Functions (from app.py) ---
def is_image_request(message: str) -> bool:
    direct = ["generate image", "create image", "make image", "draw", "show me",
              "generate", "create", "make", "paint", "sketch", "design"]
    visual = ["picture","photo","image","drawing","painting","illustration",
              "logo","poster","banner","artwork","design","concept art",
              "cat","dog","car","house","tree","flower","sunset","mountain",
              "dragon","robot","castle","forest","ocean","city","space",
              "portrait","landscape","abstract","cartoon","anime","realistic"]
    desc = ["beautiful","colorful","dark","bright","magical","mysterious","cute",
            "scary","elegant","modern","vintage","futuristic","minimalist",
            "detailed","vibrant","peaceful","dramatic"]
    m = message.lower()
    if any(t in m for t in direct):
        return True
    if any(k in m for k in visual) or (any(w in m for w in desc) and len(message.split()) <= 8):
        return True
    if re.search(r"\b(a|an)\s+\w+", m) and len(message.split()) <= 6:
        return True
    return False

def clean_prompt(message: str) -> str:
    m = message.lower()
    triggers = ["generate image of","generate image","create image of","create image",
                "make image of","make image","draw me","draw","show me","paint me",
                "generate","create","make","paint","sketch","design"]
    clean = message
    for t in triggers:
        if t in m:
            i = m.find(t)
            clean = message[:i] + message[i+len(t):]
            break
    return clean.strip() or message.strip()

def generate_image(prompt: str):
    if not HF_TOKEN:
        return None
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "guidance_scale": 7.5,
            "num_inference_steps": 4,
            "width": 1024,
            "height": 1024
        }
    }
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content))
    except Exception:
        pass
    return None

def process_image_with_watermark(image: Image.Image, prompt: str):
    try:
        wm_hash = generate_hash(prompt)
        watermarked_img = encode_watermark(image, wm_hash)
        save_watermark(wm_hash, prompt)

        buf = io.BytesIO()
        watermarked_img.save(buf, format="PNG")
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{img_data}"
    except Exception:
        return None

def detect_watermark_in_image(image: Image.Image):
    try:
        extracted_hash = decode_watermark(image)
        prompt = get_prompt_by_hash(extracted_hash)
        if prompt:
            return f"✅ This image was generated by AI with the prompt: '{prompt}'"
        else:
            return "❌ No AI watermark detected in this image."
    except Exception:
        return "❌ Could not analyze this image for watermarks."

# --- Flask Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Check if user uploaded an image for watermark detection
    if "uploaded_image" in data and data["uploaded_image"]:
        try:
            # Decode base64 image
            image_data = data["uploaded_image"].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            result = detect_watermark_in_image(image)
            return jsonify({
                "type": "text",
                "content": result
            })
        except Exception as e:
            return jsonify({
                "type": "text",
                "content": f"❌ Error analyzing image: {str(e)}"
            })

    # Check if it's an image generation request
    if is_image_request(message):
        if not HF_TOKEN:
            return jsonify({
                "type": "text",
                "content": "❌ HF_TOKEN not set. Please add your Hugging Face token to .env file."
            })

        try:
            prompt = clean_prompt(message)

            # Generate image
            image = generate_image(prompt)
            if not image:
                return jsonify({
                    "type": "text",
                    "content": "❌ Image generation failed. Please try again."
                })

            # Add watermark
            watermarked_image_data = process_image_with_watermark(image, prompt)
            if not watermarked_image_data:
                return jsonify({
                    "type": "text",
                    "content": "❌ Failed to add watermark. Please try again."
                })

            return jsonify({
                "type": "image",
                "content": f"Here's your image: \"{prompt}\"",
                "image": watermarked_image_data
            })

        except Exception as e:
            return jsonify({
                "type": "text",
                "content": f"❌ Error: {str(e)}"
            })

    # Handle regular chat responses
    fallback = "Describe an image you want, e.g., \"neon coffee shop at night\", or upload an image to check if it was AI-generated."
    quick_responses = {
        "hi": "Hi! Describe anything and I'll create it, or upload an image to check if it's AI-generated.",
        "hello": "Hello! What would you like me to make? Or upload an image for watermark detection.",
        "hey": "Hey! Ready when you are.",
        "how are you": "All good—what should we generate?",
        "thanks": "You're welcome! Want another one?",
        "thank you": "Anytime! What's next?",
    }

    response = quick_responses.get(message.lower(), fallback)
    return jsonify({
        "type": "text",
        "content": response
    })

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

# Static facts as fallback if API fails
FALLBACK_STATS = [
    "91% of people can't distinguish between deepfake videos and real ones after just a few seconds of viewing.",
    "In 2024, Trump shared AI-generated images of Taylor Swift endorsing him, which were later confirmed as fake.",
    "Deepfake detection technology lags behind creation tools by approximately 6-12 months.",
    "96% of deepfake videos online target women, with 90% being non-consensual adult content.",
    "The first detected use of deepfakes in a cyberattack was in 2019, stealing $243,000 from a UK company.",
    "Facebook removes over 1 million pieces of AI-manipulated content monthly, but experts estimate 10x more goes undetected.",
    "South Korea saw 5,000+ cases of deepfake abuse in schools in 2024, prompting national legislation.",
    "Voice cloning can now replicate someone's speech with just 3 seconds of audio, down from hours in 2020.",
    "Financial fraud using AI voice cloning increased 1,100% from 2022 to 2024.",
    "Research shows that warning labels on AI content reduce belief in fake media by only 23%.",
    "Deepfake creation apps have been downloaded over 500 million times across mobile platforms.",
    "The 2024 election saw the first widespread use of AI-generated political content in campaign ads."
]

def generate_educational_stat():
    """Generate an educational statistic about deepfakes, AI content, and misinformation."""
    if not client:
        return random.choice(FALLBACK_STATS)

    try:
        # Use Anthropic API to generate educational content
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": """Generate one fascinating, educational statistic about deepfakes, AI-generated content, misinformation, or related cybersecurity issues. Include specific numbers, percentages, or real-world examples like:
                - Recent incidents (Trump AI images, celebrity deepfakes, etc.)
                - Detection challenges and success rates
- Financial impact of AI fraud
- Platform statistics on fake content
- Regulatory responses worldwide
- Technology advancement rates

Make it concise (under 120 characters), factual, and eye-opening. Start directly with the fact, no introductions."""
            }]
        )

        stat = response.content[0].text.strip()

        # Fallback to random stat if response is too long or empty
        if len(stat) > 150 or len(stat) < 20:
            return random.choice(FALLBACK_STATS)

        return stat

    except Exception as e:
        print(f"Error generating stat: {e}")
        return random.choice(FALLBACK_STATS)

@app.route("/get_stat", methods=["GET"])
def get_stat():
    """Return a random educational statistic about AI content and deepfakes."""
    try:
        stat = generate_educational_stat()
        return jsonify({
            "stat": stat
        })
    except Exception as e:
        return jsonify({
            "error": f"Failed to generate statistic: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
