import requests
from dotenv import load_dotenv
import os
load_dotenv()
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate_image(prompt):
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        # Save output image
        with open("output.png", "wb") as f:
            f.write(response.content)
        print("Image generated → output.png")
    else:
        print("Error:", response.status_code, response.text)


generate_image("A Coffee Shop with a futuristic infra but with a clam ,clean & peaceful set up and environment with few beautiful people/couples around it .")
