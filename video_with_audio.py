import os
from moviepy.editor import ImageClip, concatenate_videoclips
from pydub import AudioSegment
import subprocess

# =======================
# CONFIG
# =======================
IMAGE_FOLDER = "generated_content"
AUDIO_INPUT = "output_60_seconds.mp3"
OUTPUT_VIDEO = "final_video_working.mp4"
DURATION_PER_IMAGE = 5

# =======================
# STEP 1 — PREPARE AUDIO
# =======================
print("🎵 Preparing audio...")

audio = AudioSegment.from_mp3(AUDIO_INPUT)

# Convert to stereo
if audio.channels == 1:
    audio = audio.set_channels(2)

# Boost volume
audio = audio + 12
audio = audio.normalize()

# Export as high-quality WAV
AUDIO_WAV = "final_audio.wav"
audio.export(AUDIO_WAV, format="wav", parameters=["-ar", "44100", "-ac", "2"])

print(f"✔ Audio ready: {AUDIO_WAV}")

# =======================
# STEP 2 — GET IMAGE FILES
# =======================
all_images = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(".png")
])

if len(all_images) < 12:
    raise Exception(f"❌ Need 12 images, found only {len(all_images)}")

images = all_images[:12]
print(f"✔ Found {len(images)} images")

# =======================
# STEP 3 — CREATE IMAGE LIST FILE FOR FFMPEG
# =======================
print("📝 Creating image list...")

with open("images_list.txt", "w") as f:
    for img in images:
        # FFmpeg concat demuxer format
        f.write(f"file '{os.path.abspath(img)}'\n")
        f.write(f"duration {DURATION_PER_IMAGE}\n")
    # Last image needs to be repeated for proper duration
    f.write(f"file '{os.path.abspath(images[-1])}'\n")

print("✔ Image list created")

# =======================
# STEP 4 — CREATE VIDEO WITH FFMPEG DIRECTLY
# =======================
print("🎬 Creating video with FFmpeg (this is the reliable way)...")

try:
    # Create video from images with audio in ONE STEP
    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "images_list.txt",
        "-i", AUDIO_WAV,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-vf", "fps=24,format=yuv420p,scale=1024:1024",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-shortest",
        OUTPUT_VIDEO
    ], capture_output=True, text=True, check=True)
    
    print("✔ Video created successfully!")
    
    # Cleanup
    os.remove("images_list.txt")
    
    print(f"\n🎉 SUCCESS! Your video: {OUTPUT_VIDEO}")
    print(f"🔊 Audio file used: {AUDIO_WAV}")
    print("\n✅ This should work in VLC, Windows Media Player, and browsers!")
    
except subprocess.CalledProcessError as e:
    print(f"❌ FFmpeg failed!")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
    
except Exception as e:
    print(f"❌ Error: {e}")