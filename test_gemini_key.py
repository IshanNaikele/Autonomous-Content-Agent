import os
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv

load_dotenv()
# --- 1. Set Your API Key ---
# IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual key.
# Alternatively, set the environment variable GEMINI_API_KEY.
GEMINI_API_KEY = "AIzaSyCBTlb29Nc2x0S5-9QYCTWLXy0iyWuZNBo"

# --- 2. Test Function ---
def test_gemini_key(api_key: str) -> bool:
    """
    Attempts a simple text generation request to validate the API key.
    
    A successful connection confirms the key is valid and the project has a quota.
    """
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("🛑 ERROR: Please replace 'YOUR_GEMINI_API_KEY' with your actual key.")
        return False

    try:
        # Initialize the client with the provided API key
        client = genai.Client(api_key=api_key)
        
        # Make a small, cheap request to a stable, general-purpose model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'Hello' in one word."
        )
        
        # Check for success
        if response.text and len(response.text.strip()) > 0:
            print("✅ SUCCESS: Gemini API Key is valid and working.")
            print(f"   Model Response: '{response.text.strip()}'")
            return True
        else:
            # This handles cases where the API returns a success status but no text (rare)
            print("⚠️ WARNING: API call succeeded but returned no content.")
            return True

    except APIError as e:
        # Catch the specific API exception for errors like invalid key, quota, or permission denial
        error_message = str(e)
        if "API key" in error_message or "AUTHENTICATION_FAILED" in error_message:
            print("❌ FAILURE: Gemini API Key is invalid or permissions are denied.")
            print(f"   Details: {error_message}")
        elif "RESOURCE_EXHAUSTED" in error_message:
            print("❌ FAILURE: Key is likely valid, but quota limit has been reached.")
            print("   Details: Please check your quota and billing status.")
        else:
            print(f"❌ FAILURE: An unexpected API error occurred: {error_message}")
        return False
    except Exception as e:
        # Catch other potential errors (network issues, incorrect SDK version, etc.)
        print(f"❌ FAILURE: An unexpected error occurred outside the API call: {e}")
        return False

# --- 3. Run the Test ---
test_gemini_key(GEMINI_API_KEY)