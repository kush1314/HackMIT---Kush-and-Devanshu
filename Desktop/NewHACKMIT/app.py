import gradio as gr
import requests
import io
from PIL import Image
import time
import re
from dotenv import load_dotenv
import os

# -------------------- Setup --------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# -------------------- Helpers --------------------
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
    r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if r.status_code == 200:
        try:
            return Image.open(io.BytesIO(r.content))
        except Exception:
            return None
    return None

def chatbot_with_images(message, history):
    if is_image_request(message):
        prompt = clean_prompt(message)
        try:
            history.append((message, "Generating…"))
            yield history, ""
            img = generate_image(prompt)
            if img:
                p = f"generated_image_{int(time.time())}.png"
                img.save(p)
                history[-1] = (message, f"Here’s your image: “{prompt}”")
                history.append((None, (p,)))
            else:
                if not HF_TOKEN:
                    history[-1] = (message, "Your HF token isn’t set. Add HF_TOKEN to your .env and restart.")
                else:
                    history[-1] = (message, "Image generation isn’t available right now. Try again.")
        except Exception as e:
            history[-1] = (message, f"Error: {str(e)}")
        yield history, ""
        return

    fallback = "Describe an image you want, e.g., “neon coffee shop at night”."
    quick = {
        "hi": "Hi! Describe anything and I’ll create it.",
        "hello": "Hello! What would you like me to make?",
        "hey": "Hey! Ready when you are.",
        "how are you": "All good—what should we generate?",
        "thanks": "You’re welcome! Want another one?",
        "thank you": "Anytime! What’s next?",
    }
    history.append((message, quick.get(message.lower(), fallback)))
    yield history, ""

# -------------------- Minimal B/W CSS (single border, no avatars) --------------------
css = """
:root{
  --black:#000;
  --white:#fff;
}

/* Page */
body, .gradio-container{
  background: var(--white) !important;
  color: var(--black) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  font-size: 16px;
  line-height: 1.55;
}

/* Layout frame */
.shell{ max-width: 920px; margin: 20px auto; padding: 0 16px; }

/* Header */
.header{
  display:flex; align-items:flex-end; justify-content:space-between;
  padding: 12px 0; border-bottom: 2px solid var(--black);
}
.title{ font-size: 24px; font-weight: 800; letter-spacing: .02em; }
.subtitle{
  font-size: 12px; font-weight: 700; text-transform: uppercase;
  border: 2px solid var(--black); padding: 4px 8px; border-radius: 999px;
}

/* Card container */
.card{ border: 2px solid var(--black); border-radius: 10px; overflow: hidden; background: var(--white); }

/* Chatbot area */
#chat{ height: 560px !important; background: var(--white) !important; border: none !important; padding: 0 !important; }

/* Remove double borders: wrappers get no borders/padding; bubble gets the only border */
#chat .message, #chat .message-wrap{
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
  box-shadow: none !important;
}

/* Hide avatars (removes the tiny black square) */
#chat .avatar, #chat img[alt="User"], #chat img[alt="Assistant"]{ display:none !important; }

/* Single bubble style */
#chat .bubble{
  border: 2px solid var(--black) !important;
  border-radius: 10px !important;
  padding: 12px 14px !important;
  max-width: 740px !important;
  background: var(--white) !important;
  color: var(--black) !important;
}

/* Invert user bubble */
#chat .message.user .bubble{
  background: var(--black) !important;
  color: var(--white) !important;
}

/* Input bar */
.input{
  display:flex; gap:10px; align-items:center;
  padding: 12px; border-top: 2px solid var(--black);
  background: var(--white);
}
.input .gr-textbox textarea, .input input[type="text"]{
  background: var(--white) !important; color: var(--black) !important;
  border: 2px solid var(--black) !important; border-radius: 8px !important;
  padding: 12px 14px !important; outline: none !important;
}
.input .gr-textbox textarea::placeholder{ color: var(--black) !important; opacity: .5; }

/* Buttons */
.btn{
  border-radius: 8px !important; border: 2px solid var(--black) !important;
  padding: 10px 14px !important; font-weight: 800 !important; letter-spacing: .01em;
  background: var(--white) !important; color: var(--black) !important;
}
.btn-primary{ background: var(--black) !important; color: var(--white) !important; }
.btn:active{ transform: translateY(1px); }

/* Footer note */
.footer{
  display:flex; justify-content:space-between; align-items:center;
  font-size: 12px; padding: 10px 0; color: var(--black);
}
.footer .pill{
  border: 2px solid var(--black); border-radius: 999px; padding: 4px 8px; font-weight: 700;
}
"""

# -------------------- App --------------------
with gr.Blocks(css=css, title="ImageBot – B/W") as demo:
    with gr.Column(elem_classes=["shell"]):
        # Header
        with gr.Row(elem_classes=["header"]):
            gr.HTML('<div class="title">ImageBot</div>')
            gr.HTML('<div class="subtitle">Black & White</div>')

        # Card (chat + input)
        with gr.Column(elem_classes=["card"]):
            # Note: bubble_full_width True reduces awkward nesting space
            chat = gr.Chatbot(
                elem_id="chat",
                bubble_full_width=True,
                show_label=False,
                show_copy_button=False
            )
            with gr.Row(elem_classes=["input"]):
                msg = gr.Textbox(
                    placeholder="Describe anything to generate…",
                    show_label=False, container=True, scale=5, autofocus=True,
                )
                send_btn = gr.Button("Send", elem_classes=["btn","btn-primary"], scale=1)
                clear_btn = gr.Button("Clear", elem_classes=["btn"], scale=1)

        # Footer
        gr.HTML(
            '<div class="footer">'
            '<div>Tip: “a futuristic city skyline at dawn”</div>'
            f'<div class="pill">HF Token: {"Set" if HF_TOKEN else "Missing"}</div>'
            '</div>'
        )

        # Logic
        def handle_message(message, history):
            if message.strip():
                yield from chatbot_with_images(message, history)

        msg.submit(handle_message, [msg, chat], [chat, msg])
        send_btn.click(handle_message, [msg, chat], [chat, msg])
        clear_btn.click(lambda: ([], ""), None, [chat, msg])

        demo.load(lambda: [("", "Hi! I’m ready. Describe an image and I’ll make it.")], None, chat)

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
