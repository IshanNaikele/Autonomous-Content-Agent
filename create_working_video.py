import os
import subprocess

print("=" * 70)
print("🎬 CREATING VIDEO WITH WORKING AUDIO - ULTIMATE FIX")
print("=" * 70)

# Configuration
IMAGE_FOLDER = "generated_content"
AUDIO_INPUT = "output_60_seconds.mp3"
OUTPUT_VIDEO = "working_video.mp4"
DURATION_PER_IMAGE = 5

# Step 1: Get images
print("\n📸 Step 1: Loading images...")
all_images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith('.png')
])[:12]

print(f"✅ Using {len(all_images)} images")
for i, img in enumerate(all_images, 1):
    print(f"   {i:2d}. {img}")

# Step 2: Create concat file
print("\n📝 Step 2: Creating image list...")
concat_file = "images.txt"
with open(concat_file, "w", encoding="utf-8") as f:
    for img in all_images:
        img_path = os.path.abspath(os.path.join(IMAGE_FOLDER, img)).replace("\\", "/")
        f.write(f"file '{img_path}'\n")
        f.write(f"duration {DURATION_PER_IMAGE}\n")
    
    # Last image
    last_img = os.path.abspath(os.path.join(IMAGE_FOLDER, all_images[-1])).replace("\\", "/")
    f.write(f"file '{last_img}'\n")

print(f"✅ Created {concat_file}")

# Step 3: Create video with PROPERLY RE-ENCODED audio
print("\n🎥 Step 3: Creating video (this will take 30-60 seconds)...")
print("⏳ Please wait...\n")

# Remove old output
if os.path.exists(OUTPUT_VIDEO):
    os.remove(OUTPUT_VIDEO)

# THE KEY FIX: Re-encode audio instead of copying
cmd = [
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_file,
    "-i", AUDIO_INPUT,
    
    # Video settings
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    
    # Audio settings - THE FIX IS HERE
    "-c:a", "aac",              # Re-encode to AAC
    "-b:a", "192k",             # High bitrate
    "-ar", "48000",             # Standard sample rate (changed from 44100)
    "-ac", "2",                 # Force stereo
    "-af", "pan=stereo|c0=c0|c1=c0,volume=3.0",  # Convert mono to stereo properly + boost
    
    # Other settings
    "-shortest",
    "-movflags", "+faststart",
    
    OUTPUT_VIDEO
]

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    
    print("✅ VIDEO CREATED!\n")
    print("=" * 70)
    
    # Get info
    file_size = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
    print(f"📁 File: {os.path.abspath(OUTPUT_VIDEO)}")
    print(f"📦 Size: {file_size:.2f} MB")
    print(f"⏱️  Duration: {len(all_images) * DURATION_PER_IMAGE} seconds")
    print("=" * 70)
    
    # Verify streams
    print("\n🔍 Verifying output...")
    verify_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,channels,sample_rate",
        "-of", "default=noprint_wrappers=1",
        OUTPUT_VIDEO
    ]
    
    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
    print(verify_result.stdout)
    
    if "codec_type=audio" in verify_result.stdout and "channels=2" in verify_result.stdout:
        print("✅ Video has STEREO audio!")
    
    # Cleanup
    os.remove(concat_file)
    
    print("\n🎉 SUCCESS!")
    print("\n📺 NOW TEST IT:")
    print("   1. Close all media players")
    print(f"   2. Double-click: {OUTPUT_VIDEO}")
    print("   3. Or drag into VLC")
    print("   4. YOU SHOULD HEAR AUDIO NOW!")
    
    print("\n💡 IF STILL NO AUDIO:")
    print("   - Make sure system volume is UP")
    print("   - Try with headphones")
    print("   - Try opening in Chrome browser")
    print(f"   - Play the original MP3 to confirm speakers work")
    
except subprocess.CalledProcessError as e:
    print("❌ ERROR!")
    print("\nSTDERR:")
    print(e.stderr)
    
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "=" * 70)