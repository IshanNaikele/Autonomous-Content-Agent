import os
import subprocess

# =======================
# CONFIG
# =======================
IMAGE_FOLDER = "generated_content"
AUDIO_INPUT = "output_60_seconds.mp3"
OUTPUT_VIDEO = "final_working_video.mp4"
DURATION_PER_IMAGE = 5

print("🔍 Step 1: Checking your images...")

# Get all PNG images
all_images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(".png")
])

if len(all_images) < 12:
    print(f"❌ Error: Need 12 images, found only {len(all_images)}")
    exit(1)

images = all_images[:12]
print(f"✔ Found {len(images)} images")

# =======================
# STEP 2 — CREATE CONCAT FILE
# =======================
print("\n📝 Step 2: Creating image list...")

concat_file = "concat_list.txt"
with open(concat_file, "w", encoding="utf-8") as f:
    for img in images:
        img_path = os.path.join(IMAGE_FOLDER, img).replace("\\", "/")
        f.write(f"file '{img_path}'\n")
        f.write(f"duration {DURATION_PER_IMAGE}\n")
    # Repeat last image
    last_img = os.path.join(IMAGE_FOLDER, images[-1]).replace("\\", "/")
    f.write(f"file '{last_img}'\n")

print("✔ Image list created")

# =======================
# STEP 3 — TEST AUDIO FILE
# =======================
print("\n🔊 Step 3: Testing audio file...")

if not os.path.exists(AUDIO_INPUT):
    print(f"❌ Error: Audio file '{AUDIO_INPUT}' not found!")
    exit(1)

# Check audio with ffprobe
result = subprocess.run([
    "ffprobe", "-v", "error",
    "-show_entries", "stream=codec_name,channels,sample_rate",
    "-of", "default=noprint_wrappers=1",
    AUDIO_INPUT
], capture_output=True, text=True)

print("Audio info:")
print(result.stdout)

# =======================
# STEP 4 — CREATE VIDEO (SIMPLEST METHOD)
# =======================
print("\n🎬 Step 4: Creating video with FFmpeg...")
print("This may take 30-60 seconds...\n")

cmd = [
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_file,
    "-i", AUDIO_INPUT,
    "-vsync", "vfr",
    "-pix_fmt", "yuv420p",
    "-vf", "scale=1024:1024",
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-c:a", "aac",
    "-b:a", "192k",
    "-strict", "experimental",
    "-shortest",
    OUTPUT_VIDEO
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    print("✅ SUCCESS!")
    print(f"\n🎉 Video created: {OUTPUT_VIDEO}")
    print(f"📂 Location: {os.path.abspath(OUTPUT_VIDEO)}")
    
    # Cleanup
    if os.path.exists(concat_file):
        os.remove(concat_file)
    
    print("\n" + "="*50)
    print("HOW TO PLAY:")
    print("1. Right-click the video file")
    print("2. Choose 'Open with' → VLC Media Player")
    print("3. If no audio, press Ctrl+Up Arrow in VLC to boost volume")
    print("="*50)
    
except subprocess.CalledProcessError as e:
    print("❌ FFmpeg ERROR!")
    print("\nError output:")
    print(e.stderr)
    print("\n💡 Troubleshooting:")
    print("1. Make sure all PNG files are valid images")
    print("2. Make sure output_60_seconds.mp3 plays in a media player")
    print("3. Try running: ffmpeg -version")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")