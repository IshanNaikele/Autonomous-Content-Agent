from elevenlabs.client import ElevenLabs
from httpx import HTTPStatusError
import os 
from dotenv import load_dotenv

load_dotenv()


# Initialize client
ElevenLabs_api_key=os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=ElevenLabs_api_key)

try:
    print("Fetching voices...")
    voices = client.voices.get_all()
    print("Voices fetched successfully!")
    print(voices)

except HTTPStatusError as http_err:
    print("HTTP error from ElevenLabs API:", http_err)

except Exception as err:
    print("Some other error:", err)
