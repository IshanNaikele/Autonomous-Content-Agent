import os
from moviepy.editor import ImageClip, concatenate_videoclips
from pydub import AudioSegment
import subprocess

# =======================
# CONFIG
# =======================
IMAGE_FOLDER = "generated_content"
AUDIO_INPUT = "output_60_seconds.mp3"
OUTPUT_VIDEO = "final_video_FIXED.mp4"
DURATION_PER_IMAGE = 5

# =======================
# STEP 1 — FIX AUDIO (CONVERT TO WAV FOR COMPATIBILITY)
# =======================
print("🎵 Converting audio to WAV format...")

audio = AudioSegment.from_mp3(AUDIO_INPUT)

# Force stereo
if audio.channels == 1:
    audio = audio.set_channels(2)

# Boost volume
audio = audio + 15

# Normalize
audio = audio.normalize()

# Export as WAV (most compatible)
AUDIO_WAV = "audio_fixed.wav"
audio.export(AUDIO_WAV, format="wav", parameters=["-ar", "44100"])

print(f"✔ Audio converted to WAV")

# =======================
# STEP 2 — LOAD IMAGES
# =======================
all_images = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(".png")
])

if len(all_images) < 12:
    raise Exception(f"❌ Need 12 images, found only {len(all_images)}")

images = all_images[:12]
print(f"✔ Loaded {len(images)} images")

# =======================
# STEP 3 — CREATE SILENT VIDEO
# =======================
print("🎬 Creating video...")

clips = [ImageClip(img).set_duration(DURATION_PER_IMAGE) for img in images]
slideshow = concatenate_videoclips(clips, method="compose")

temp_video = "temp_silent.mp4"
slideshow.write_videofile(
    temp_video, 
    fps=24, 
    codec="libx264", 
    audio=False,
    verbose=False,
    logger=None
)

print("✔ Silent video created")

# =======================
# STEP 4 — MERGE WITH PROPER SETTINGS
# =======================
print("🔊 Merging video and audio with FFmpeg...")

try:
    # This command ensures maximum compatibility
    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", AUDIO_WAV,
        "-c:v", "libx264",          # Re-encode video
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",              # AAC audio codec
        "-b:a", "192k",             # Audio bitrate
        "-ar", "44100",             # Sample rate
        "-ac", "2",                 # Stereo
        "-pix_fmt", "yuv420p",      # Pixel format for compatibility
        "-movflags", "+faststart",  # Web optimization
        "-shortest",
        OUTPUT_VIDEO
    ], check=True, capture_output=True, text=True)
    
    print("✔ Video merged successfully!")
    
    # Cleanup
    os.remove(temp_video)
    os.remove(AUDIO_WAV)
    
    print(f"\n🎉 DONE! Video saved: {OUTPUT_VIDEO}")
    print("\n📋 Try playing it with:")
    print("   1. VLC Media Player (recommended)")
    print("   2. Windows Media Player")
    print("   3. Any modern browser (drag & drop)")
    
except subprocess.CalledProcessError as e:
    print(f"❌ FFmpeg Error:")
    print(e.stderr)

except Exception as e:
    print(f"❌ Error: {e}")