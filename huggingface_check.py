import requests
import os
import json

# --- 1. Configuration ---
# Your target model API URL
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

# It's best practice to load the token from the environment, as you already do.
# Make sure your terminal or execution environment has the HF_TOKEN variable set.
HF_TOKEN = os.getenv("HF_TOKEN") 

# --- 2. Request Setup ---
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Simple test prompt for basic functionality check
payload = {
    "inputs": "A bright red minimalistic robot checking its system status on a futuristic blue screen, digital art, ultra-detailed.",
    "parameters": {
        "num_inference_steps": 4,  # Use minimum steps for a quick test
        "height": 512,
        "width": 512
    }
}

# --- 3. Execution ---
def check_hf_access():
    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN environment variable not set.")
        return

    print(f"Attempting connection to: {API_URL}")
    print("-" * 40)
    
    try:
        # We expect a POST request for image generation
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=30)
        
        if response.status_code == 200:
            print("✅ SUCCESS: API Token is valid and the model is running.")
            print(f"   Response status: {response.status_code} (OK)")
            
            # The response content will be a binary image file.
            # We check the first few bytes to confirm it's an image.
            content_type = response.headers.get("Content-Type", "N/A")
            content_length = len(response.content) / 1024 # KB
            print(f"   Received Content-Type: {content_type}")
            print(f"   Received Content Size: {content_length:.2f} KB (Expected binary image data)")
            
            # Optional: Save the image to confirm content is correct
            with open("hf_status_check.png", "wb") as f:
                f.write(response.content)
            print("   Test image saved as hf_status_check.png")
            
        elif response.status_code == 401:
            print("❌ FAILURE (401 Unauthorized): HF_TOKEN is invalid or expired.")
        elif response.status_code == 404:
            print("❌ FAILURE (404 Not Found): Model URL is incorrect or model is not available via this endpoint.")
        elif response.status_code == 429:
            print("❌ FAILURE (429 Too Many Requests): Rate limit reached. Try again later or check your plan.")
        elif 400 <= response.status_code < 500:
            print(f"❌ FAILURE ({response.status_code}): Client-side error. Detail: {response.text}")
        else:
            print(f"❌ UNEXPECTED FAILURE ({response.status_code}): Server error. Detail: {response.text}")

    except requests.exceptions.Timeout:
        print("❌ FAILURE: Request timed out. The model may be loading (cold start) or under heavy load.")
    except Exception as e:
        print(f"❌ AN UNEXPECTED ERROR OCCURRED: {e}")

if __name__ == "__main__":
    check_hf_access()