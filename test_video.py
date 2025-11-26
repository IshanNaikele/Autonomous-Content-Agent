import subprocess
import os

print("🔍 DIAGNOSING YOUR VIDEO/AUDIO ISSUE\n")
print("=" * 70)

# Files to check
VIDEO_FILE = "final_with_audio.mp4"
AUDIO_FILE = "output_60_seconds.mp3"

# Test 1: Check if files exist
print("\n📁 TEST 1: Checking if files exist...")
if os.path.exists(VIDEO_FILE):
    size = os.path.getsize(VIDEO_FILE) / (1024 * 1024)
    print(f"✅ Video exists: {VIDEO_FILE} ({size:.2f} MB)")
else:
    print(f"❌ Video NOT found: {VIDEO_FILE}")
    
if os.path.exists(AUDIO_FILE):
    size = os.path.getsize(AUDIO_FILE) / (1024 * 1024)
    print(f"✅ Audio exists: {AUDIO_FILE} ({size:.2f} MB)")
else:
    print(f"❌ Audio NOT found: {AUDIO_FILE}")

# Test 2: Check video streams
print("\n🎬 TEST 2: Checking video streams...")
cmd = [
    "ffprobe", "-v", "error",
    "-show_entries", "stream=codec_type,codec_name,channels,sample_rate",
    "-of", "default=noprint_wrappers=1",
    VIDEO_FILE
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    
    if "codec_type=audio" in result.stdout:
        print("✅ Audio stream IS present in video")
    else:
        print("❌ Audio stream MISSING from video")
        
    if "codec_type=video" in result.stdout:
        print("✅ Video stream IS present")
        
except Exception as e:
    print(f"❌ Error checking video: {e}")

# Test 3: Play audio file separately
print("\n🎵 TEST 3: Can you hear this audio file?")
print(f"   Opening: {AUDIO_FILE}")
print("   ⚠️  Listen carefully - if you hear voice, audio works!")

try:
    # Try to open the audio file
    if os.name == 'nt':  # Windows
        os.startfile(AUDIO_FILE)
    print("   ✅ Audio file opened. Do you hear it? (Y/N)")
except Exception as e:
    print(f"   ❌ Could not auto-open: {e}")
    print(f"   Manually open: {os.path.abspath(AUDIO_FILE)}")

# Test 4: Extract audio from video
print("\n🔊 TEST 4: Extracting audio from video to test...")
EXTRACTED_AUDIO = "extracted_audio.mp3"

extract_cmd = [
    "ffmpeg", "-y", "-i", VIDEO_FILE,
    "-vn",  # No video
    "-acodec", "copy",  # Copy audio stream
    EXTRACTED_AUDIO
]

try:
    subprocess.run(extract_cmd, check=True, capture_output=True)
    print(f"✅ Extracted audio to: {EXTRACTED_AUDIO}")
    print(f"   Now opening it...")
    
    if os.name == 'nt':
        os.startfile(EXTRACTED_AUDIO)
        
    print("\n   ⚠️  LISTEN TO THIS FILE!")
    print("   If you hear audio here but NOT in video, it's a player issue")
    
except Exception as e:
    print(f"❌ Could not extract audio: {e}")

# Summary
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print("\n1. If you heard the original MP3 → Audio source is fine")
print("2. If you heard the extracted MP3 → Video HAS audio")
print("3. If video plays silently → Player issue or volume")
print("\n💡 NEXT STEPS:")
print("   a) Try playing in different player (VLC, Chrome browser)")
print("   b) Check system volume / mute status")
print("   c) Try headphones to rule out speaker issues")
print("   d) If still fails, we'll try different encoding")
print("=" * 70)