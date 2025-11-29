"""
Quick test file to verify Fireworks AI image generation works
Run: python test_fireworks_image.py
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

if not FIREWORKS_API_KEY:
    print("❌ ERROR: FIREWORKS_API_KEY not found in .env file")
    print("Please add: FIREWORKS_API_KEY=your_key_here")
    exit(1)

print("=" * 70)
print("🔥 TESTING FIREWORKS AI IMAGE GENERATION")
print("=" * 70)

# Correct endpoint from documentation
url = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"

headers = {
    "Content-Type": "application/json",
    "Accept": "image/jpeg",
    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
}

# Simple test prompt
data = {
    "prompt": "a cute robot mascot waving, photorealistic, 8K, studio lighting"
}

print(f"\n📝 Prompt: {data['prompt']}")
print(f"🔗 Endpoint: {url}")
print(f"⏳ Sending request...\n")

try:
    response = requests.post(url, headers=headers, json=data, timeout=60)
    
    print(f"📡 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS! Image received")
        
        # Save the image
        output_filename = "fireworks_test_image.jpg"
        with open(output_filename, "wb") as f:
            f.write(response.content)
        
        # Get file size
        file_size = len(response.content) / 1024
        
        print(f"💾 Image saved as: {output_filename}")
        print(f"📦 File size: {file_size:.1f} KB")
        print(f"\n{'=' * 70}")
        print(f"🎉 TEST PASSED! Fireworks AI is working!")
        print(f"{'=' * 70}")
        print(f"\n💡 To view the image, run:")
        print(f"   start {output_filename}  (Windows)")
        print(f"   open {output_filename}   (Mac)")
        
    elif response.status_code == 401:
        print("❌ ERROR 401: Unauthorized")
        print("Your API key is invalid or expired")
        print("Check your FIREWORKS_API_KEY in .env file")
        
    elif response.status_code == 402:
        print("❌ ERROR 402: Payment Required")
        print("Your free credits are exhausted")
        print("Add credits at: https://fireworks.ai/account/billing")
        
    elif response.status_code == 429:
        print("❌ ERROR 429: Rate Limit")
        print("Too many requests, wait a moment and try again")
        
    else:
        print(f"❌ ERROR {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ ERROR: Request timed out (60 seconds)")
    print("Try again in a moment")
    
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Connection failed")
    print("Check your internet connection")
    
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 70)