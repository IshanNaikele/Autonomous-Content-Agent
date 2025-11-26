import os
import subprocess

# =======================
# CONFIG
# =======================
IMAGE_FOLDER = "generated_content"
AUDIO_INPUT = "output_60_seconds.mp3"
OUTPUT_VIDEO = "final_video_WITH_AUDIO.mp4"
DURATION_PER_IMAGE = 5

print("🔍 Step 1: Finding images...")

# Get all PNG images and sort them
all_images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(".png")
])

print(f"✔ Found {len(all_images)} total images")

# Take the first 12 images
if len(all_images) < 12:
    print(f"⚠ Warning: Only {len(all_images)} images found, using all of them")
    images = all_images
else:
    images = all_images[:12]
    print(f"✔ Using first 12 images")

# Print which images will be used
print("\n📸 Images to be used:")
for i, img in enumerate(images, 1):
    print(f"  {i}. {img}")

# =======================
# STEP 2 — CREATE CONCAT FILE
# =======================
print("\n📝 Step 2: Creating FFmpeg input file...")

concat_file = "ffmpeg_images.txt"
with open(concat_file, "w", encoding="utf-8") as f:
    for img in images:
        # Use forward slashes and absolute path
        img_path = os.path.abspath(os.path.join(IMAGE_FOLDER, img)).replace("\\", "/")
        f.write(f"file '{img_path}'\n")
        f.write(f"duration {DURATION_PER_IMAGE}\n")
    
    # Repeat last image (FFmpeg requirement)
    last_img = os.path.abspath(os.path.join(IMAGE_FOLDER, images[-1])).replace("\\", "/")
    f.write(f"file '{last_img}'\n")

print(f"✔ Created {concat_file}")

# =======================
# STEP 3 — CREATE VIDEO WITH AUDIO
# =======================
print("\n🎬 Step 3: Creating video with audio...")
print("⏳ This will take 30-60 seconds...\n")

cmd = [
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_file,
    "-i", AUDIO_INPUT,
    "-vsync", "vfr",
    "-pix_fmt", "yuv420p",
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "23",
    "-c:a", "aac",
    "-b:a", "192k",
    "-ar", "44100",
    "-ac", "2",  # Force stereo
    "-af", "volume=2.0",  # Boost volume by 2x
    "-shortest",
    OUTPUT_VIDEO
]

try:
    # Run FFmpeg
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        check=True
    )
    
    print("✅ SUCCESS! Video created!\n")
    print("="*60)
    print(f"📁 Video location: {os.path.abspath(OUTPUT_VIDEO)}")
    print(f"🎵 Audio: Stereo, 192kbps, boosted 2x volume")
    print(f"🎞️  Video: {len(images)} images × {DURATION_PER_IMAGE}s = {len(images)*DURATION_PER_IMAGE}s total")
    print("="*60)
    
    # Cleanup
    if os.path.exists(concat_file):
        os.remove(concat_file)
    
    print("\n🎉 HOW TO PLAY:")
    print("1. Open VLC Media Player")
    print(f"2. Drag '{OUTPUT_VIDEO}' into VLC")
    print("3. If audio is still quiet, press Ctrl+Up Arrow to boost volume")
    print("\n💡 If VLC shows errors, check that all PNG files are valid images")
    
except subprocess.CalledProcessError as e:
    print("❌ ERROR! FFmpeg failed to create video\n")
    print("STDERR output:")
    print(e.stderr)
    print("\n" + "="*60)
    print("🔧 TROUBLESHOOTING:")
    print("1. Check if images are corrupted:")
    print(f"   Open a few PNG files from '{IMAGE_FOLDER}' manually")
    print("2. Verify audio plays:")
    print(f"   Double-click '{AUDIO_INPUT}' to play it")
    print("3. Make sure FFmpeg is working:")
    print("   Run: ffmpeg -version")
    print("="*60)
    
except FileNotFoundError:
    print("❌ ERROR! FFmpeg not found in system PATH")
    print("Make sure FFmpeg is installed and added to PATH")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")