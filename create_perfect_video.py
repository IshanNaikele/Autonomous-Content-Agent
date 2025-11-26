import os
import subprocess

print("=" * 70)
print("🎬 CREATING VIDEO WITH WORKING AUDIO - FIXED IMAGE TIMING")
print("=" * 70)

# Configuration
IMAGE_FOLDER = "generated_content"
AUDIO_INPUT = "output_60_seconds.mp3"
OUTPUT_VIDEO = "perfect_video.mp4"
DURATION_PER_IMAGE = 5

# Step 1: Get images
print("\n📸 Step 1: Loading images...")
all_images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith('.png')
])[:12]

print(f"✅ Using {len(all_images)} images")
for i, img in enumerate(all_images, 1):
    print(f"   {i:2d}. {img} → {(i-1)*5}s to {i*5}s")

total_duration = len(all_images) * DURATION_PER_IMAGE
print(f"\n⏱️  Total video duration: {total_duration} seconds")

# Step 2: Create concat file with FIXED format
print("\n📝 Step 2: Creating image list (FIXED FORMAT)...")
concat_file = "images.txt"

with open(concat_file, "w", encoding="utf-8") as f:
    for i, img in enumerate(all_images):
        img_path = os.path.abspath(os.path.join(IMAGE_FOLDER, img)).replace("\\", "/")
        f.write(f"file '{img_path}'\n")
        
        # IMPORTANT: Add duration for ALL images including the last one
        f.write(f"duration {DURATION_PER_IMAGE}\n")
    
    # FFmpeg quirk: Repeat the last image without duration
    last_img = os.path.abspath(os.path.join(IMAGE_FOLDER, all_images[-1])).replace("\\", "/")
    f.write(f"file '{last_img}'\n")

print(f"✅ Created {concat_file}")

# Debug: Show what's in the file
print("\n🔍 Concat file preview (first 6 lines):")
with open(concat_file, "r") as f:
    for i, line in enumerate(f):
        if i < 6:
            print(f"   {line.strip()}")

# Step 3: Create video with proper timing
print("\n🎥 Step 3: Creating video (30-60 seconds)...")
print("⏳ Please wait...\n")

# Remove old output
if os.path.exists(OUTPUT_VIDEO):
    os.remove(OUTPUT_VIDEO)

# FFmpeg command with vsync cfr for consistent frame timing
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
    "-r", "25",  # Set framerate explicitly
    "-vsync", "cfr",  # Constant frame rate (important for timing!)
    
    # Audio settings
    "-c:a", "aac",
    "-b:a", "192k",
    "-ar", "48000",
    "-ac", "2",
    "-af", "pan=stereo|c0=c0|c1=c0,volume=3.0",
    
    # Timing
    "-t", str(total_duration),  # Explicitly set duration
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
    print(f"⏱️  Duration: {total_duration} seconds ({len(all_images)} images × {DURATION_PER_IMAGE}s)")
    print("=" * 70)
    
    # Verify duration
    print("\n🔍 Verifying video duration...")
    duration_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        OUTPUT_VIDEO
    ]
    
    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
    actual_duration = float(duration_result.stdout.strip())
    print(f"✅ Actual duration: {actual_duration:.2f} seconds")
    
    if abs(actual_duration - total_duration) < 1:
        print("✅ Duration is correct!")
    else:
        print(f"⚠️  Duration mismatch: expected {total_duration}s, got {actual_duration:.2f}s")
    
    # Verify audio
    print("\n🔍 Verifying audio...")
    verify_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,channels",
        "-of", "default=noprint_wrappers=1",
        OUTPUT_VIDEO
    ]
    
    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
    if "codec_type=audio" in verify_result.stdout and "channels=2" in verify_result.stdout:
        print("✅ Stereo audio confirmed!")
    
    # Cleanup
    os.remove(concat_file)
    
    print("\n🎉 ALL PERFECT!")
    print("\n📺 TEST YOUR VIDEO:")
    print(f"   Command: start {OUTPUT_VIDEO}")
    print("   Or drag into VLC/Browser")
    print("\n✅ Check that:")
    print("   - All 12 images appear (5 seconds each)")
    print("   - Images start immediately at 0:00")
    print("   - Audio plays throughout")
    print("   - Total duration is 60 seconds")
    
except subprocess.CalledProcessError as e:
    print("❌ ERROR!")
    print("\nSTDERR:")
    print(e.stderr)
    
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "=" * 70)
print("\n💡 TO PLAY: start perfect_video.mp4")
print("=" * 70)