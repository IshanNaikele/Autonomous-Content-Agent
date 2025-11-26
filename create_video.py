import os
import subprocess
from pathlib import Path

# =======================
# CONFIGURATION
# =======================
IMAGE_FOLDER = "generated_content"
AUDIO_INPUT = "output_60_seconds.mp3"
OUTPUT_VIDEO = "final_with_audio.mp4"
DURATION_PER_IMAGE = 5  # seconds

print("=" * 70)
print("🎬 VIDEO CREATOR WITH AUDIO (FIXED FOR MONO)")
print("=" * 70)

# =======================
# STEP 1: VERIFY FILES EXIST
# =======================
print("\n📋 Step 1: Checking files...")

if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ ERROR: Folder '{IMAGE_FOLDER}' not found!")
    exit(1)

if not os.path.exists(AUDIO_INPUT):
    print(f"❌ ERROR: Audio file '{AUDIO_INPUT}' not found!")
    exit(1)

print(f"✅ Image folder found: {IMAGE_FOLDER}")
print(f"✅ Audio file found: {AUDIO_INPUT}")

# =======================
# STEP 2: CHECK AUDIO FORMAT
# =======================
print("\n🎵 Step 2: Analyzing audio...")

probe_cmd = [
    "ffprobe", "-v", "error",
    "-show_entries", "stream=channels,sample_rate,codec_name",
    "-of", "default=noprint_wrappers=1",
    AUDIO_INPUT
]

try:
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    print(f"Audio info:\n{result.stdout}")
except:
    print("⚠️  Could not probe audio, continuing anyway...")

# =======================
# STEP 3: GET IMAGES
# =======================
print("\n📸 Step 3: Loading images...")

all_images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if len(all_images) == 0:
    print(f"❌ ERROR: No images found in '{IMAGE_FOLDER}'!")
    exit(1)

# Use first 12 images
images_to_use = all_images[:12]
print(f"✅ Found {len(all_images)} images, using {len(images_to_use)}")

# Show which images will be used
for i, img in enumerate(images_to_use, 1):
    print(f"   {i:2d}. {img}")

# =======================
# STEP 4: CREATE INPUT FILE FOR FFMPEG
# =======================
print("\n📝 Step 4: Preparing FFmpeg input...")

input_file = "input_list.txt"
with open(input_file, "w", encoding="utf-8") as f:
    for img in images_to_use:
        # Get absolute path and convert backslashes to forward slashes
        img_path = os.path.abspath(os.path.join(IMAGE_FOLDER, img)).replace("\\", "/")
        f.write(f"file '{img_path}'\n")
        f.write(f"duration {DURATION_PER_IMAGE}\n")
    
    # Add last image one more time (FFmpeg requirement)
    last_img = os.path.abspath(os.path.join(IMAGE_FOLDER, images_to_use[-1])).replace("\\", "/")
    f.write(f"file '{last_img}'\n")

print(f"✅ Created '{input_file}'")

# =======================
# STEP 5: CREATE VIDEO WITH FFMPEG (FIXED AUDIO)
# =======================
print("\n🎥 Step 5: Creating video with audio...")
print("⏳ This may take 30-90 seconds, please wait...\n")

# Delete old output if it exists
if os.path.exists(OUTPUT_VIDEO):
    os.remove(OUTPUT_VIDEO)
    print(f"🗑️  Removed old '{OUTPUT_VIDEO}'")

# FFmpeg command - FIXED to handle mono audio properly
cmd = [
    "ffmpeg",
    "-y",  # Overwrite output file
    "-f", "concat",
    "-safe", "0",
    "-i", input_file,
    "-i", AUDIO_INPUT,
    "-c:v", "libx264",  # Video codec
    "-preset", "medium",  # Encoding speed
    "-crf", "23",  # Quality
    "-pix_fmt", "yuv420p",  # Pixel format for compatibility
    "-c:a", "aac",  # Audio codec
    "-b:a", "192k",  # Audio bitrate
    "-ar", "44100",  # Audio sample rate
    # REMOVED: "-ac", "2"  # Don't force stereo - let FFmpeg handle it
    "-af", "volume=3.0",  # Boost volume by 3x
    "-shortest",  # Stop when shortest input ends
    "-movflags", "+faststart",  # Optimize for streaming
    OUTPUT_VIDEO
]

try:
    # Run FFmpeg with visible output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    
    print("✅ VIDEO CREATED SUCCESSFULLY!\n")
    print("=" * 70)
    print(f"📁 Output: {os.path.abspath(OUTPUT_VIDEO)}")
    
    # Check file size
    file_size = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)  # MB
    print(f"📦 Size: {file_size:.2f} MB")
    
    print(f"⏱️  Duration: {len(images_to_use) * DURATION_PER_IMAGE} seconds")
    print(f"🎵 Audio: AAC, 192 kbps, boosted 3x")
    print(f"🎞️  Video: H.264, {len(images_to_use)} images")
    print("=" * 70)
    
    # Verify audio is in the output
    print("\n🔍 Verifying audio in output...")
    verify_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name",
        "-of", "default=noprint_wrappers=1",
        OUTPUT_VIDEO
    ]
    
    try:
        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, check=True)
        output_info = verify_result.stdout
        
        if "codec_type=audio" in output_info:
            print("✅ Audio stream confirmed in video!")
        else:
            print("⚠️  Warning: No audio stream detected in output")
            
        print(f"\nOutput streams:\n{output_info}")
    except:
        print("⚠️  Could not verify output, but video was created")
    
    # Cleanup
    if os.path.exists(input_file):
        os.remove(input_file)
        print(f"\n🧹 Cleaned up '{input_file}'")
    
    print("\n🎉 ALL DONE!")
    print("\n📺 TO TEST THE VIDEO:")
    print(f"   1. Open VLC Media Player")
    print(f"   2. Drag '{OUTPUT_VIDEO}' into VLC")
    print(f"   3. Press SPACE to play")
    print(f"   4. Press Ctrl+Up Arrow if volume is low")
    
    print("\n💡 TROUBLESHOOTING:")
    print("   - Still no audio? Try playing in different player")
    print("   - Try playing in browser (drag & drop into Chrome/Edge)")
    print("   - Check your system volume isn't muted")
    print("   - Try headphones to verify audio works")

except subprocess.CalledProcessError as e:
    print("\n❌ FFMPEG ERROR!\n")
    print("STDERR:")
    print(e.stderr)
    print("\n" + "=" * 70)
    print("🔧 DEBUGGING INFO:")
    print("   - Check if images are valid (open a few manually)")
    print("   - Verify audio file plays separately")
    print("   - Try the simple_video.py script instead")
    print("=" * 70)

except FileNotFoundError:
    print("\n❌ ERROR: FFmpeg not found!")
    print("Make sure FFmpeg is installed and in your PATH")

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")

print("\n" + "=" * 70)