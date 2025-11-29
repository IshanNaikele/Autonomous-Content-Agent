"""
FREE Subtitle Generator for Your Video Pipeline
Converts script text → SRT subtitle file → Burns into video
"""

import os
import re
from typing import List, Tuple
import subprocess


def estimate_word_timings(text: str, total_duration: float) -> List[Tuple[str, float, float]]:
    """
    Estimates word-by-word timings based on total duration.
    Returns list of (word, start_time, end_time) tuples.
    
    Args:
        text: The full script text
        total_duration: Total audio duration in seconds
    
    Returns:
        List of (word, start_time, end_time)
    """
    # Clean and split text into words
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return []
    
    # Calculate average time per word
    time_per_word = total_duration / total_words
    
    word_timings = []
    current_time = 0.0
    
    for word in words:
        start_time = current_time
        end_time = current_time + time_per_word
        word_timings.append((word, start_time, end_time))
        current_time = end_time
    
    return word_timings


def create_subtitle_chunks(word_timings: List[Tuple[str, float, float]], 
                          words_per_subtitle: int = 5) -> List[Tuple[str, float, float]]:
    """
    Groups words into subtitle chunks (5-8 words per line).
    Returns list of (text, start_time, end_time).
    """
    chunks = []
    
    for i in range(0, len(word_timings), words_per_subtitle):
        chunk_words = word_timings[i:i + words_per_subtitle]
        
        if not chunk_words:
            continue
        
        # Combine words
        text = ' '.join([w[0] for w in chunk_words])
        start_time = chunk_words[0][1]
        end_time = chunk_words[-1][2]
        
        chunks.append((text, start_time, end_time))
    
    return chunks


def format_srt_time(seconds: float) -> str:
    """
    Converts seconds to SRT time format: HH:MM:SS,mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_file(text: str, duration: float, output_path: str = "subtitles.srt") -> str:
    """
    Generates an SRT subtitle file from script text.
    
    Args:
        text: The script text
        duration: Total audio duration in seconds
        output_path: Where to save the SRT file
    
    Returns:
        Path to the generated SRT file
    """
    print(f"\n🎬 Generating subtitles for {duration}s video...")
    
    # Get word timings
    word_timings = estimate_word_timings(text, duration)
    
    # Create subtitle chunks (5-8 words per line looks best)
    chunks = create_subtitle_chunks(word_timings, words_per_subtitle=6)
    
    print(f"✅ Created {len(chunks)} subtitle segments")
    
    # Write SRT file
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (subtitle_text, start, end) in enumerate(chunks, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
            f.write(f"{subtitle_text}\n")
            f.write("\n")
    
    print(f"✅ Subtitle file saved: {output_path}")
    return output_path


def burn_subtitles_into_video(video_path: str, 
                              srt_path: str, 
                              output_path: str,
                              style: str = "netflix") -> str:
    """
    Burns subtitles into video using FFmpeg (permanent, Netflix-style).
    
    Args:
        video_path: Input video file
        srt_path: SRT subtitle file
        output_path: Output video with burned subtitles
        style: 'netflix', 'youtube', or 'minimal'
    
    Returns:
        Path to output video
    """
    print(f"\n🔥 Burning subtitles into video (style: {style})...")
    
    # Different subtitle styles
    styles = {
        "netflix": "FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=1,MarginV=40",
        "youtube": "FontName=Roboto,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=4,Outline=3,Shadow=2,MarginV=50,Bold=1",
        "minimal": "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=30"
    }
    
    subtitle_style = styles.get(style, styles["netflix"])
    
    # Convert SRT path for FFmpeg (handle Windows paths)
    srt_path_ffmpeg = srt_path.replace("\\", "/").replace(":", "\\:")
    
    # FFmpeg command to burn subtitles
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles={srt_path_ffmpeg}:force_style='{subtitle_style}'",
        "-c:a", "copy",  # Copy audio without re-encoding
        "-preset", "medium",
        "-crf", "23",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Subtitled video created: {output_path} ({file_size:.2f} MB)")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e.stderr}")
        raise


def add_subtitles_to_video(video_path: str, 
                           script_text: str, 
                           duration: float,
                           style: str = "netflix") -> str:
    """
    Complete pipeline: Script → SRT → Burn into video.
    
    Args:
        video_path: Path to video without subtitles
        script_text: The narration script
        duration: Video duration in seconds
        style: Subtitle style ('netflix', 'youtube', 'minimal')
    
    Returns:
        Path to final video with subtitles
    """
    print("\n" + "="*70)
    print("📝 ADDING SUBTITLES TO VIDEO")
    print("="*70)
    
    # Generate SRT file
    video_dir = os.path.dirname(video_path)
    srt_path = os.path.join(video_dir, "subtitles.srt")
    generate_srt_file(script_text, duration, srt_path)
    
    # Create output filename
    base_name = os.path.splitext(video_path)[0]
    output_path = f"{base_name}_with_subtitles.mp4"
    
    # Burn subtitles
    burn_subtitles_into_video(video_path, srt_path, output_path, style)
    
    print("\n" + "="*70)
    print("✅ SUBTITLES ADDED SUCCESSFULLY!")
    print("="*70)
    print(f"📁 Original: {video_path}")
    print(f"📁 With Subtitles: {output_path}")
    
    return output_path


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Add subtitles to an existing video
    sample_script = """
    Hey there. Ever wondered why your morning routine feels so chaotic? 
    Here's the truth. You're not broken. Your system is. 
    In the next thirty seconds, I'm going to show you three micro-habits 
    that will transform your entire day. No fluff. No BS. Just results. 
    Ready? Let's dive in.
    """
    
    # Test with a sample video (replace with your actual video path)
    video_file = "final_video.mp4"
    
    if os.path.exists(video_file):
        subtitled_video = add_subtitles_to_video(
            video_path=video_file,
            script_text=sample_script,
            duration=15.0,  # 15 seconds
            style="netflix"  # Try 'youtube' or 'minimal' too
        )
        print(f"\n🎉 Done! Play your subtitled video:")
        print(f"   start {subtitled_video}")
    else:
        print(f"❌ Video file not found: {video_file}")
        print("Update the video_file path to test subtitle generation.")