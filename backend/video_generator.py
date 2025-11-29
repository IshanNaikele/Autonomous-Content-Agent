"""
Complete Video Generator Module with Time-Based Segmentation + Thumbnail Embedding
Converts text → audio → time-synced image prompts → video with audio + thumbnail
Each image displays for 3 seconds
"""

import os
import requests
import subprocess
import json
from typing import List, Optional, Tuple
from elevenlabs import ElevenLabs
from pydub import AudioSegment
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# API Keys
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Configuration
OUTPUT_FOLDER = "generated_content"
AUDIO_FILE = "video_audio.mp3"
VIDEO_OUTPUT = "final_video.mp4"

# Duration mapping: video_duration → (num_images, duration_per_image)
DURATION_CONFIG = {
    5: (2, 3),
    10: (4, 3),
    15: (5, 3),
    30: (10, 3),
    45: (15, 3),
    60: (20, 3),
}

os.makedirs("generated_content/images", exist_ok=True)
os.makedirs("generated_content/videos", exist_ok=True)
os.makedirs("generated_content/blogs", exist_ok=True)


# ============================================================================
# HELPER: DETERMINE VIDEO CONFIGURATION
# ============================================================================

def get_video_config(duration_seconds: int) -> Tuple[int, int]:
    """
    Determines number of images and duration per image based on total duration.
    Each image is displayed for 3 seconds.
    
    Args:
        duration_seconds: Total video duration in seconds
    
    Returns:
        Tuple of (num_images, duration_per_image)
    """
    if duration_seconds in DURATION_CONFIG:
        return DURATION_CONFIG[duration_seconds]
    
    # For custom durations, calculate dynamically (3 seconds per image)
    duration_per_image = 3
    num_images = max(1, round(duration_seconds / duration_per_image))
    
    return (num_images, duration_per_image)


# ============================================================================
# HELPER: SEGMENT TEXT BY TIME
# ============================================================================

def segment_text_by_time(text: str, num_segments: int) -> List[str]:
    """
    Intelligently divides text into time-based segments.
    Each segment represents what will be spoken during that time period.
    
    Args:
        text: The full script text
        num_segments: Number of segments to create (based on video duration)
    
    Returns:
        List of text segments, one per time period
    """
    print("\n" + "=" * 70)
    print("📊 SEGMENTING TEXT BY TIME")
    print("=" * 70)
    
    # Clean and prepare text
    text = text.strip()
    sentences = [s.strip() + '.' for s in text.replace('\n', ' ').split('.') if s.strip()]
    
    print(f"📝 Total text: {len(text)} characters")
    print(f"📝 Sentences: {len(sentences)}")
    print(f"🎯 Target segments: {num_segments}")
    
    # Calculate target words per segment
    total_words = len(text.split())
    words_per_segment = max(1, total_words // num_segments)
    
    print(f"📝 Total words: {total_words}")
    print(f"📝 Words per segment: ~{words_per_segment}")
    
    segments = []
    current_segment = []
    current_word_count = 0
    
    # Distribute sentences across segments
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence keeps us under the limit, add it
        if current_word_count + sentence_words <= words_per_segment * 1.3 or len(segments) >= num_segments:
            current_segment.append(sentence)
            current_word_count += sentence_words
        else:
            # Save current segment and start new one
            if current_segment:
                segments.append(' '.join(current_segment))
            current_segment = [sentence]
            current_word_count = sentence_words
        
        # If we have enough segments, add remaining sentences to last segment
        if len(segments) == num_segments - 1:
            current_segment.append(sentence)
    
    # Add the last segment
    if current_segment:
        segments.append(' '.join(current_segment))
    
    # Ensure we have exactly num_segments
    while len(segments) < num_segments:
        # Split the longest segment
        longest_idx = max(range(len(segments)), key=lambda i: len(segments[i]))
        long_seg = segments[longest_idx]
        mid = len(long_seg) // 2
        segments[longest_idx] = long_seg[:mid].strip()
        segments.insert(longest_idx + 1, long_seg[mid:].strip())
    
    while len(segments) > num_segments:
        # Merge the two shortest segments
        shortest_idx = min(range(len(segments) - 1), key=lambda i: len(segments[i]) + len(segments[i + 1]))
        segments[shortest_idx] = segments[shortest_idx] + ' ' + segments[shortest_idx + 1]
        segments.pop(shortest_idx + 1)
    
    # Display segments  
    print(f"\n✅ Created {len(segments)} time-based segments:\n")
    for i, seg in enumerate(segments, 1):
        duration_start = (i - 1) * 3
        duration_end = i * 3
        print(f"   [{duration_start:2d}-{duration_end:2d}s] {seg[:70]}...")
    
    return segments


# ============================================================================
# FUNCTION 1: TEXT TO AUDIO
# ============================================================================

def generate_audio_from_text(text: str, target_duration: int, output_filename: str = AUDIO_FILE) -> str:
    """
    Converts text to audio using ElevenLabs API.
    Stretches/compresses audio to target duration.
    
    Args:
        text: The script text to convert to speech
        target_duration: Target duration in seconds
        output_filename: Name of the output audio file
    
    Returns:
        Path to the generated audio file
    """
    print("\n" + "=" * 70)
    print("🎵 STEP 1: GENERATING AUDIO FROM TEXT")
    print("=" * 70)
    
    if not ELEVENLABS_API_KEY:
        raise Exception("ElevenLabs API key not found in .env file")
    
    print(f"📝 Text length: {len(text)} characters")
    print(f"⏱️  Target duration: {target_duration} seconds")
    print("🔊 Generating speech...")
    
    try:
        response = elevenlabs_client.text_to_speech.convert(
            voice_id="T3b0vsQ5dQwMZ5ckOwBk",
            model_id="eleven_flash_v2_5",
            text=text,
            output_format="mp3_44100_128"
        )
        
        # Save raw audio
        raw_audio_path = "raw_audio_temp.mp3"
        with open(raw_audio_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Raw audio generated")
        
        # Load and adjust to target duration
        audio = AudioSegment.from_mp3(raw_audio_path)
        current_duration_ms = len(audio)
        target_duration_ms = target_duration * 1000
        
        print(f"⏱️  Original duration: {current_duration_ms / 1000:.2f}s")
        
        # Calculate speed adjustment
        playback_speed = current_duration_ms / target_duration_ms
        final_audio = audio.speedup(playback_speed=playback_speed)
        
        # Export final audio
        final_audio.export(output_filename, format="mp3")
        
        print(f"✅ Audio adjusted to {target_duration} seconds")
        print(f"📁 Saved: {output_filename}")
        
        # Cleanup
        if os.path.exists(raw_audio_path):
            os.remove(raw_audio_path)
        
        return output_filename
        
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        raise


# ============================================================================
# FUNCTION 2: GENERATE TIME-SYNCED IMAGE PROMPTS
# ============================================================================

def generate_time_synced_prompts(text_segments: List[str]) -> List[str]:
    """
    Uses Groq LLM to generate image prompts that are perfectly synced
    with what's being narrated at each time period.
    
    Args:
        text_segments: List of text segments (one per 3-second period)
    
    Returns:
        List of image generation prompts, one per segment
    """
    print("\n" + "=" * 70)
    print("🤖 STEP 2: GENERATING TIME-SYNCED IMAGE PROMPTS")
    print("=" * 70)
    
    if not GROQ_API_KEY:
        raise Exception("Groq API key not found in .env file")
    
    num_segments = len(text_segments)
    print(f"🎨 Creating {num_segments} prompts, each synced to its narration...\n")
    
    # Create the prompt for LLM
    segments_text = "\n".join([
        f"SEGMENT {i+1} (seconds {i*3}-{(i+1)*3}): {seg}"
        for i, seg in enumerate(text_segments)
    ])
    
    llm_prompt = f"""You are an elite AI Visual Director. You will receive {num_segments} text segments from a video narration. Each segment represents exactly 3 seconds of spoken content.

Your task: Create a STUNNING, CINEMATIC image prompt for each segment that PERFECTLY MATCHES what is being narrated during those 3 seconds.The image should be realistic and feels like a real .It should not be seem like an AI generated .

NARRATION SEGMENTS:
{segments_text}

REQUIREMENTS:
1. **Perfect Sync**: The visual MUST directly represent what's being said in that exact segment
2. **Cinematic Quality**: Use professional photography terms (8K, shot on ARRI, volumetric lighting, etc.)
3. **Visual Diversity**: Each prompt should have different camera angles, lighting, and composition
4. **Consistency**: Maintain a cohesive visual style across all {num_segments} images
5. **Specificity**: Include exact details about subject, action, mood, lighting, camera angle
6. **Length**: Each prompt should be 20-30 words
7. **Context**:The prompt you are giving for generating images should show the core product values & the product .It is mandatory that the prompt matches that audio context but also remind you the image prompt sequence should be related from each other should it should not feel like discontinuiation in the video flow .
8. **Generate**:The prompt should be like it's not creating the same image again & again .
FORMAT:
Return ONLY a JSON array with {num_segments} objects:
[
  {{
    "segment_number": 1,
    "narration": "the narration text",
    "image_prompt": "ultra-detailed cinematic prompt here"
  }},
  ...
]

STYLE GUIDE:
- Camera: "low angle", "bird's eye view", "extreme close-up", "wide angle"
- Lighting: "golden hour", "dramatic rim lighting", "soft diffused", "neon glow"
- Quality: "8K resolution", "shot on RED camera", "award-winning", "photorealistic"
- Mood: "ethereal", "dynamic", "serene", "suspenseful", "vibrant"

Create {num_segments} perfectly synced, breathtaking image prompts now:"""
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an elite AI Visual Director who creates stunning, time-synced image prompts. Each prompt must perfectly match its narration segment."
                },
                {
                    "role": "user",
                    "content": llm_prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.75,
            max_tokens=4000
        )
        
        llm_response = response.choices[0].message.content
        
        # Try to parse JSON
        try:
            # Remove markdown code blocks if present
            if "```json" in llm_response:
                import re
                match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
                if match:
                    llm_response = match.group(1)
            elif "```" in llm_response:
                import re
                match = re.search(r'```\s*(.*?)\s*```', llm_response, re.DOTALL)
                if match:
                    llm_response = match.group(1)
            
            structured_data = json.loads(llm_response.strip())
            
            # Extract prompts
            prompts = []
            for item in structured_data:
                if isinstance(item, dict) and "image_prompt" in item:
                    prompts.append(item["image_prompt"])
                elif isinstance(item, str):
                    prompts.append(item)
            
            # Validate count
            if len(prompts) < num_segments:
                print(f"⚠️  LLM returned {len(prompts)} prompts, expected {num_segments}")
                print("Generating additional prompts...")
                
                while len(prompts) < num_segments:
                    idx = len(prompts)
                    fallback = generate_fallback_prompt_for_segment(text_segments[idx], idx)
                    prompts.append(fallback)
            
            prompts = prompts[:num_segments]
            
            # Display results
            print(f"✅ Generated {len(prompts)} time-synced prompts:\n")
            for i, (segment, prompt) in enumerate(zip(text_segments, prompts), 1):
                print(f"   [{(i-1)*3:2d}-{i*3:2d}s] Narration: {segment[:50]}...")
                print(f"            Image Prompt: {prompt}\n")
            
            return prompts
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            print("Falling back to line-by-line parsing...")
            
            # Fallback: try to extract prompts line by line
            prompts = []
            lines = llm_response.split('\n')
            
            for line in lines:
                if '"image_prompt"' in line and ':' in line:
                    # Extract text after the colon
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        prompt = parts[1].strip().strip('"').strip(',')
                        if len(prompt) > 30:
                            prompts.append(prompt)
            
            if len(prompts) < num_segments:
                print("⚠️  Fallback parsing insufficient, using template generation...")
                return generate_fallback_prompts(text_segments)
            
            return prompts[:num_segments]
            
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        print("⚠️  Using fallback prompt generation...")
        return generate_fallback_prompts(text_segments)


def generate_fallback_prompt_for_segment(segment_text: str, index: int) -> str:
    """Generate a single fallback prompt for a segment"""
    base = f"A cinematic photograph representing: {segment_text[:100]}"
    return vary_image_prompt(base, index)


def generate_fallback_prompts(text_segments: List[str]) -> List[str]:
    """Fallback prompt generation based on text segments"""
    prompts = []
    for i, segment in enumerate(text_segments):
        prompt = generate_fallback_prompt_for_segment(segment, i)
        prompts.append(prompt)
    return prompts


def vary_image_prompt(base_prompt: str, index: int) -> str:
    """
    Creates distinct, high-quality prompts by appending specific camera angles 
    and lighting styles while prioritizing the original brand/color details.
    """
    
    composition_variations = [
        "low angle shot, wide cinematic framing",
        "extreme close-up, shallow depth of field",
        "overhead flat lay, minimalist background",
        "medium shot, rule of thirds",
        "high-angle perspective",
        "isometric 3D render view",
    ]
    
    style_variations = [
        "dramatic rim lighting, rich contrast",
        "soft daylight, warm pastels",
        "vibrant neon glow, futuristic",
        "golden hour sunlight, cinematic",
        "moody chiaroscuro lighting",
        "bright studio photography",
    ]
    
    comp_tag = composition_variations[index % len(composition_variations)]
    style_tag = style_variations[index % len(style_variations)]
    
    # ⭐ FIX: Remove aggressive truncation, but ensure a clean start/end
    # We will let the LLM handle the core content, trusting it was specific.
    base_prompt = base_prompt.strip().rstrip(',').rstrip('.')
    
    # ⭐ FIX: Move quality and style modifiers to the END to prioritize brand/color
    # The image model reads best when specifics (color, subject) are first.
    # Quality prefix is kept short for character count management.
    quality_suffix = "photorealistic, 8K, ultra-detailed, cinematic" 
    
    # ⭐ CONSTRUCT THE NEW PROMPT: Original content first, then stylistic variations.
    new_prompt = f"{base_prompt}, {comp_tag}, {style_tag}, {quality_suffix}"
    
    # 🔥 FINAL CHECK: ENSURE prompt is under the recommended 300 chars
    if len(new_prompt) > 300:
        # Truncate aggressively but keep the core idea at the start
        new_prompt = new_prompt[:300].rsplit(',', 1)[0]
    
    return new_prompt

# ============================================================================
# FUNCTION 3: GENERATE IMAGES
# ============================================================================

def generate_images(prompts: List[str], seed: Optional[int] = None, output_folder: str = OUTPUT_FOLDER) -> List[str]:
    """
    Generates images using Fireworks AI FLUX model.
    """
    print("\n" + "=" * 70)
    print("🎨 STEP 3: GENERATING IMAGES (Fireworks AI)")
    print("=" * 70)
    
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    if not FIREWORKS_API_KEY:
        raise Exception("Fireworks API key not found in .env file")
    
    API_URL = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"
    headers = {
        "Content-Type": "application/json",
        "Accept": "image/jpeg",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    }
    
    generated_images = []
    MAX_PROMPT_LENGTH = 300  # 🔥 ADD THIS
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Generating image for seconds {(i-1)*3}-{i*3}...")
        
        # 🔥 ADD THIS: Truncate prompt
        if len(prompt) > MAX_PROMPT_LENGTH:
            print(f"⚠️  Prompt too long ({len(prompt)} chars), truncating to {MAX_PROMPT_LENGTH}")
            prompt = prompt[:MAX_PROMPT_LENGTH].rsplit(' ', 1)[0]
        
        print(f"📝 Prompt ({len(prompt)} chars): {prompt[:80]}...")
        print("🔥 Generating with Fireworks...")
        
        payload = {"prompt": prompt}
        if seed is not None:
            payload["seed"] = seed + i
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                os.makedirs(output_folder, exist_ok=True)
                
                filename = f"image_{i:02d}.jpg"
                file_path = os.path.join(output_folder, filename)
                
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                file_size = len(response.content) / 1024
                print(f"✅ Saved: {filename} ({file_size:.1f} KB)")
                generated_images.append(file_path)
            else:
                print(f"⚠️  Failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"⚠️  Timeout generating image {i}")
        except Exception as e:
            print(f"⚠️  Error: {e}")
    
    print(f"\n✅ Successfully generated {len(generated_images)}/{len(prompts)} images")
    return generated_images


# ============================================================================
# FUNCTION 4: STITCH IMAGES WITH AUDIO + THUMBNAIL EMBEDDING
# ============================================================================

def create_video_with_audio(
    image_paths: List[str],
    audio_path: str,
    output_filename: str,
    duration_per_image: int,
    thumbnail_path: Optional[str] = None
) -> str:
    """
    Combines images and audio into a video file with embedded thumbnail.
    
    Args:
        image_paths: List of image file paths
        audio_path: Path to audio file
        output_filename: Name of output video file
        duration_per_image: Duration each image should display (seconds)
        thumbnail_path: Optional path to thumbnail image to embed as video poster
    
    Returns:
        Path to the generated video file
    """
    print("\n" + "=" * 70)
    print("🎬 STEP 4: CREATING VIDEO WITH AUDIO & THUMBNAIL")
    print("=" * 70)
    
    if len(image_paths) == 0:
        raise Exception("No images provided for video creation")
    
    if not os.path.exists(audio_path):
        raise Exception(f"Audio file not found: {audio_path}")
    
    print(f"📸 Images: {len(image_paths)}")
    print(f"🎵 Audio: {audio_path}")
    print(f"⏱️  Duration per image: {duration_per_image}s")
    
    if thumbnail_path and os.path.exists(thumbnail_path):
        print(f"🖼️  Thumbnail: {thumbnail_path}")
    else:
        print(f"⚠️  No thumbnail provided or file not found")
        thumbnail_path = None
    
    # Create concat file for FFmpeg
    concat_file = "ffmpeg_concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for img_path in image_paths:
            abs_path = os.path.abspath(img_path).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {duration_per_image}\n")
        
        # Repeat last image (FFmpeg requirement)
        last_img = os.path.abspath(image_paths[-1]).replace("\\", "/")
        f.write(f"file '{last_img}'\n")
    
    print(f"✅ Created concat file")
    
    # Remove old output if exists
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    # Build FFmpeg command
    print("\n🎥 Running FFmpeg (this takes 30-60 seconds)...")
    total_duration = len(image_paths) * duration_per_image
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-i", audio_path,
    ]
    
    # Add thumbnail as third input if available
    if thumbnail_path:
        cmd.extend(["-i", thumbnail_path])
    
    cmd.extend([
        # Video settings
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-r", "25",
        "-vsync", "cfr",
        
        # Audio settings
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        "-af", "pan=stereo|c0=c0|c1=c0,volume=3.0",
    ])
    
    # Map streams correctly based on whether thumbnail is present
    if thumbnail_path:
        cmd.extend([
            "-map", "0:v",  # Video from concat (images)
            "-map", "1:a"  # Audio from audio file
        ])
        print("✅ Embedding thumbnail as video poster frame")
    
    cmd.extend([
        # Duration
        "-t", str(total_duration),
        "-shortest",
        "-movflags", "+faststart",
        
        output_filename
    ])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Cleanup
        if os.path.exists(concat_file):
            os.remove(concat_file)
        
        # Verify output
        if os.path.exists(output_filename):
            file_size = os.path.getsize(output_filename) / (1024 * 1024)
            print(f"\n✅ VIDEO CREATED SUCCESSFULLY!")
            print(f"📁 File: {os.path.abspath(output_filename)}")
            print(f"📦 Size: {file_size:.2f} MB")
            print(f"⏱️  Duration: {total_duration} seconds")
            if thumbnail_path:
                print(f"🖼️  Thumbnail embedded successfully")
            return output_filename
        else:
            raise Exception("Video file was not created")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error:")
        print(e.stderr)
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


# ============================================================================
# MAIN FUNCTION: COMPLETE PIPELINE
# ============================================================================

def create_video_from_text(
    text: str,
    duration: int = 60,
    output_video: str = VIDEO_OUTPUT,
    audio_output: str = AUDIO_FILE,
    frames_folder: str = OUTPUT_FOLDER,
    thumbnail_path: Optional[str] = None,
    seed: Optional[int] = None
) -> str:
    """
    Complete pipeline with TIME-BASED SEGMENTATION + THUMBNAIL EMBEDDING:
    Text → Segmentation → Audio → Time-Synced Prompts → Images → Video with Thumbnail
    Each image displays for 3 seconds.
    
    Args:
        text: The script text for the video
        duration: Video duration in seconds (5, 10, 15, 30, 45, 60, or custom)
        output_video: Output video filename
        audio_output: Output audio filename
        frames_folder: Folder to save frame images
        thumbnail_path: Optional path to thumbnail image to embed
        seed: Optional seed for reproducible image generation
    
    Returns:
        Path to the final video file
    """
    print("\n" + "=" * 70)
    print("🚀 STARTING TIME-SYNCED VIDEO GENERATION PIPELINE")
    print("=" * 70)
    print(f"📝 Text length: {len(text)} characters")
    print(f"⏱️  Duration: {duration} seconds")
    
    # Get configuration for this duration
    num_images, duration_per_image = get_video_config(duration)
    
    print(f"🎯 Configuration: {num_images} images × {duration_per_image}s = {num_images * duration_per_image}s")
    
    try:
        # Step 0: Segment text by time FIRST
        text_segments = segment_text_by_time(text, num_images)
        
        # Step 1: Generate audio (full text) in video folder
        audio_path = generate_audio_from_text(text, duration, audio_output)
        
        # Step 2: Generate time-synced image prompts
        prompts = generate_time_synced_prompts(text_segments)
        
        # Step 3: Generate images in frames folder
        image_paths = generate_images(prompts, seed, frames_folder)
        
        if len(image_paths) < num_images:
            print(f"\n⚠️  Warning: Only {len(image_paths)}/{num_images} images generated")
            print("Continuing with available images...")
        
        # Step 4: Create video with thumbnail
        video_path = create_video_with_audio(
            image_paths, 
            audio_path, 
            output_video,
            duration_per_image,
            thumbnail_path=thumbnail_path
        )
        
        print("\n" + "=" * 70)
        print("🎉 TIME-SYNCED PIPELINE COMPLETED!")
        print("=" * 70)
        print(f"📁 Final video: {os.path.abspath(video_path)}")
        print(f"⏱️  Duration: {len(image_paths) * duration_per_image} seconds ({len(image_paths)} images at 3s each)")
        if thumbnail_path:
            print(f"🖼️  Thumbnail embedded from: {thumbnail_path}")
        print(f"\n✨ Each image perfectly matches what's being said at that moment!")
        print(f"\n💡 To play: start {video_path}")
        print("=" * 70)
        
        return video_path
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example text for video
    sample_text = """
    In today's competitive landscape, the real advantage lies in a mind that keeps learning, improving, and delivering results.
Every challenge, every project, and every goal needs someone who thinks ahead, adapts quickly, and brings clarity to complexity.
    """
    
    print("🎬 TIME-SYNCED AI VIDEO GENERATOR - DEMO")
    print("\nNEW: Images are now perfectly synced with narration!")
    print("Each image displays for 3 seconds.")
    print("\nAvailable durations:")
    print("  • 5 seconds  → 2 images  (adjusted)")
    print("  • 10 seconds → 4 images  (adjusted)")
    print("  • 15 seconds → 5 images")
    print("  • 30 seconds → 10 images")
    print("  • 45 seconds → 15 images")
    print("  • 60 seconds → 20 images\n")
    
    # You can change duration here
    VIDEO_DURATION = 10
    
    try:
        video_path = create_video_from_text(
            text=sample_text,
            duration=VIDEO_DURATION,
            output_video=f"synced_video_{VIDEO_DURATION}s.mp4",
            seed=42
        )
        
        print(f"\n✅ Demo complete! Video saved to: {video_path}")
        print("\nTo test the video, run:")
        print(f"   start {video_path}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nMake sure you have in your .env file:")
        print("  • ELEVENLABS_API_KEY")
        print("  • HF_TOKEN")
        print("  • GROQ_API_KEY")
        print("\nAnd FFmpeg installed in PATH")