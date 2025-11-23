# video.py
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import base64
import io
import os
import requests

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set")

client = InferenceClient(token=HF_TOKEN)

SVD_API_URL = "https://router.huggingface.co/models/stabilityai/stable-video-diffusion-img2vid"

class Prompt(BaseModel):
    prompt: str

@app.post("/text_to_video_free")
def text_to_video_free(data: Prompt):
    # ----------------------
    # STEP 1 – TEXT → IMAGE
    # ----------------------
    try:
        image = client.text_to_image(
            prompt=data.prompt,
            model="black-forest-labs/FLUX.1-dev"
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

    except Exception as e:
        return {"error": f"FLUX image generation failed: {e}"}

    # ----------------------
    # STEP 2 – IMAGE → VIDEO
    # ----------------------
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        files = {"file": ("image.png", img_bytes, "image/png")}

        resp = requests.post(SVD_API_URL, headers=headers, files=files)

        if resp.status_code != 200:
            return {"error": f"SVD failed: {resp.text}"}

        video_bytes = resp.content

    except Exception as e:
        return {"error": f"SVD video generation failed: {e}"}

    encoded = base64.b64encode(video_bytes).decode()
    return {"video_base64": encoded}
