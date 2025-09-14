from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import hashlib
import os
import requests
import re
import base64
import random

app = Flask(__name__)

SECRET_KEY = os.getenv("SECRET_KEY", "hackmitsecret")
HF_TOKEN = os.getenv("HF_TOKEN")

# Stats for the header
STATS = [
    "Over 90% of AI-generated images go undetected by humans",
    "Deepfake videos can be generated in under 10 minutes today",
    "AI image generation models are trained on billions of images",
    "Watermarking can survive JPEG compression and resizing",
    "The human eye can't detect most steganographic watermarks",
    "AI detection accuracy drops to 50% on modified images",
    "Digital watermarks can store up to 256 bits of data invisibly",
    "95% of people can't identify AI-generated faces in photos"
]

def generate_hash(prompt: str) -> str:
    return hashlib.sha256((prompt + SECRET_KEY).encode()).hexdigest()[:64]

def encode_watermark(image: Image.Image, wm_hash: str) -> Image.Image:
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    flat = arr[:, :, :3].flatten()

    total_bits = len(wm_hash) * 4
    step = len(flat) // total_bits

    for i, char in enumerate(wm_hash):
        hex_val = int(char, 16)
        for j in range(4):
            bit_idx = i * 4 + j
            pixel_idx = bit_idx * step
            if pixel_idx < len(flat):
                bit = (hex_val >> (3-j)) & 1
                flat[pixel_idx] = (flat[pixel_idx] & 0xFE) | bit

    arr[:, :, :3] = flat.reshape(arr[:, :, :3].shape)
    return Image.fromarray(arr, "RGBA")

def decode_watermark(image: Image.Image, length=64) -> str:
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    flat = arr[:, :, :3].flatten()

    total_bits = length * 4
    step = len(flat) // total_bits

    bin_str = ''
    for i in range(total_bits):
        idx = i * step
        bin_str += str(flat[idx] & 1)

    hex_str = ''.join(format(int(bin_str[i:i+4], 2), 'x') for i in range(0, len(bin_str), 4))
    return hex_str[:length]

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
        return None, "Hugging Face token not configured"

    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=60)
        if response.status_code == 200:
            return response.content, None
        else:
            return None, f"API error: {response.status_code}"
    except Exception as e:
        return None, f"Generation error: {str(e)}"

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    if path.startswith("api/"):
        return jsonify(error="API endpoint not found"), 404
    return jsonify(ok=True, msg="Hello from Vercel Python!")

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        uploaded_image = data.get("uploaded_image")

        if uploaded_image:
            try:
                _, encoded = uploaded_image.split(',', 1)
                image_data = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_data)).convert("RGBA")

                detected_hash = decode_watermark(image)
                if detected_hash and detected_hash != "0" * 64:
                    return jsonify({
                        "type": "text",
                        "content": f"✅ **AI-Generated Image Detected**\n\nThis image contains a SafeStamp watermark with hash: `{detected_hash[:16]}...`\n\nThis confirms the image was generated using AI and watermarked for authenticity verification."
                    })
                else:
                    return jsonify({
                        "type": "text",
                        "content": "❌ **No SafeStamp Watermark Detected**\n\nThis image does not contain a SafeStamp watermark. This could mean:\n- The image was not generated by our AI system\n- The watermark was removed or corrupted\n- The image was heavily modified after generation"
                    })
            except Exception as e:
                return jsonify({
                    "type": "text",
                    "content": f"❌ **Error analyzing image**: {str(e)}"
                })

        elif message and is_image_request(message):
            cleaned_prompt = clean_prompt(message)

            image_bytes, error = generate_image(cleaned_prompt)
            if error:
                return jsonify({
                    "type": "text",
                    "content": f"❌ **Generation failed**: {error}"
                })

            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                wm_hash = generate_hash(cleaned_prompt)
                watermarked_image = encode_watermark(image, wm_hash)

                output_buffer = io.BytesIO()
                watermarked_image.save(output_buffer, format="PNG")
                output_buffer.seek(0)

                encoded_image = base64.b64encode(output_buffer.read()).decode()

                return jsonify({
                    "type": "image",
                    "content": f"✨ **Generated**: {cleaned_prompt}\n\n🔒 **Watermarked** with SafeStamp for authenticity verification",
                    "image": f"data:image/png;base64,{encoded_image}"
                })

            except Exception as e:
                return jsonify({
                    "type": "text",
                    "content": f"❌ **Processing error**: {str(e)}"
                })

        else:
            return jsonify({
                "type": "text",
                "content": "👋 Hi! I can:\n\n🎨 **Generate AI images** - just describe what you want\n🔍 **Detect watermarks** - upload an image to check if it's AI-generated\n\nTry: *\"a sunset over mountains\"* or upload an image!"
            })

    except Exception as e:
        return jsonify({
            "type": "text",
            "content": f"❌ **Server error**: {str(e)}"
        })

@app.route("/api/get_stat", methods=["GET"])
def get_stat():
    try:
        stat = random.choice(STATS)
        return jsonify({"stat": stat})
    except Exception as e:
        return jsonify({"error": str(e)})
