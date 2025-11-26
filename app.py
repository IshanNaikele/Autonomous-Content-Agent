from elevenlabs import ElevenLabs
from pydub import AudioSegment
import math
import os 
from dotenv import load_dotenv
load_dotenv()
API_KEY =  os.getenv("ELEVENLABS_API_KEY")

client = ElevenLabs(api_key=API_KEY)

text = """
n today's fast-moving digital world, attention is the most valuable currency. 
Every idea, every message, and every story is competing to be heard. 
But the people who win are the ones who express their ideas clearly, confidently, and creatively. 
That is why content creation has become a real superpower. 
With the right tools, anyone can turn a simple thought into something meaningful— 
a video, a message, a design, or a voice that connects with thousands. 

Technology is no longer just a helper; 
it is a partner that allows you to bring your imagination to life. 
Whether you want to inspire people, educate them, or share your journey, 
your ideas deserve the best possible expression. 

So slow down for a moment, breathe, and give your creativity the space it needs. 
This is your time to create something powerful, something original, 
and something that truly represents who you are. 
Your voice matters—let it be heard.
"""

# Step 1 — Generate audio
response = client.text_to_speech.convert(
    voice_id="2EiwWnXFnvU5JabPnv8n",
    model_id="eleven_flash_v2_5",
    text=text,
    output_format="mp3_44100_128"
)

# Save original audio
with open("raw_audio.mp3", "wb") as f:
    for chunk in response:
        if chunk:
            f.write(chunk)

# Step 2 — Stretch/compress to exactly 60 seconds
audio = AudioSegment.from_mp3("raw_audio.mp3")

target_duration_ms = 60 * 1000  # 60 sec in ms
current_duration_ms = len(audio)

playback_speed = current_duration_ms / target_duration_ms

final_audio = audio.speedup(playback_speed=playback_speed)

# Step 3 — Save final 60 sec file
final_audio.export("output_60_seconds.mp3", format="mp3")

print("🎉 DONE! Saved as output_60_seconds.mp3 (exactly 60 seconds)")
