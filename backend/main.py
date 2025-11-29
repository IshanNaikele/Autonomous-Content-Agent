# backend/main.py (FINAL CODE - Image Variation and Quality Optimized + Thumbnail & Folder Fixes)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, PositiveInt
from typing import Literal, Optional, List
import os
import json
import time 
import requests
import random
from datetime import datetime

# --- API CLIENTS ---
from groq import Groq
from google import genai
from tavily import TavilyClient
from google.genai import types 
from .video_generator import create_video_from_text
 
# --- 1. Pydantic Schemas for Input and Output ---

class ContentRequest(BaseModel):
    topic_idea: str = Field(..., max_length=500, description="The main subject or idea for the content.")
    primary_format: Literal['Blog Post', 'Image', 'Video', 'Campaign'] = Field(..., description="The main deliverable format.")
    
    item_count: Optional[PositiveInt] = Field(None, description="The number of items required (must be >= 1).")
    word_count: Optional[PositiveInt] = Field(None, description="Required word count for Blog Post (must be >= 1).")
    video_duration_seconds: Optional[PositiveInt] = Field(None, description="Required duration for Video format in seconds (must be >= 1).")
    
    target_audience: str = Field(..., description="Persona of the primary reader/viewer.")
    business_goal: Literal['Awareness', 'Lead Generation', 'Conversion', 'Engagement'] = Field(..., description="The goal this content must achieve.")
    brand_tone: Literal['Expert & Formal', 'Friendly & Witty', 'Casual & Rebellious'] = Field(..., description="The required tone of voice.")
    brand_color_hex: Optional[str] = Field(None, description="Primary brand color (e.g., #007bff)")
    # ⭐ NEW OPTIONAL FIELDS FOR SUBTITLES
    enable_subtitles: bool = Field(default=True, description="Add subtitles to video")
    subtitle_style: Literal['netflix', 'youtube', 'minimal'] = Field(
        default='netflix',
        description="Subtitle visual style"
    )

    # ⭐ NEW: User tier for logo control
    user_tier: Literal['free', 'premium'] = Field(
        default='free',
        description="User subscription tier. Free users get watermarked videos."
    )

class ResearchData(BaseModel):
    target_keywords: List[str] = Field(..., description="3-5 inferred high-value keywords.")
    top_competitor_urls: List[str] = Field(..., description="The top 5 URLs found by the search engine.")
    content_gap_analysis: str = Field(..., description="The unique angle/topic the competition missed.")


class CreativeBlueprint(BaseModel):
    blog_text_prompt: str = Field(..., description="The single, complete prompt for the Blog/Text generation agent.")
    image_prompt: str = Field(..., description="The single, complete prompt for the Image generation agent.")
    video_prompt: str = Field(..., description="The single, complete prompt for the Video generation agent.")
    
# Create necessary folder structure
os.makedirs("generated_content/images", exist_ok=True)
os.makedirs("generated_content/videos", exist_ok=True)
os.makedirs("generated_content/blogs", exist_ok=True)

# --- 2. API CLIENT INITIALIZATION ---
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not all([GROQ_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY, HF_TOKEN]):
        print("One or more API keys (GROQ_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY, HF_TOKEN) are missing from environment variables.")
        groq_client, gemini_client, tavily_client = None, None, None
    else:
        groq_client = Groq(api_key=GROQ_API_KEY)
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

except Exception as e:
    print(f"FATAL ERROR: API clients failed to initialize unexpectedly. {e}")
    groq_client, gemini_client, tavily_client = None, None, None

# --- 3. CORE PHASE 2 FUNCTIONS ---

def research_and_gap_analysis(request: ContentRequest) -> ResearchData:
    """Steps 2.1 & 2.2: Uses Tavily for search, then Gemini for structured analysis."""
    
    if not gemini_client or not tavily_client:
        raise HTTPException(status_code=500, detail="API clients not initialized.")

    print("\n--- DEBUG: STEP 2.1 & 2.2 STARTING (Tavily/Gemini) ---")
    
    # --- 3A. TAVILY SEARCH ---
    tavily_query = f"Current trends, top competitors, and common questions about: {request.topic_idea} for {request.target_audience}."
    
    try:
        tavily_results = tavily_client.search(
            query=tavily_query,
            search_depth="advanced",
            max_results=2,
            include_raw_content=True
        )
    except Exception as e:
        print(f"DEBUG: Tavily search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tavily Search Failed: {e}")

    search_snippets = "\n---\n".join([r['content'] for r in tavily_results['results']])
    top_urls = [r['url'] for r in tavily_results['results']]
    print(f"DEBUG: Tavily retrieved {len(top_urls)} URLs.")

    # --- 3B. GEMINI ANALYSIS (Gap Identification & Synthesis) ---
    gemini_prompt = f"""
You are a strategic analyst. Generate a valid JSON object only, with no additional text, markdown, or code fences.

Based on the search snippets below and these strategy inputs:
- AUDIENCE: {request.target_audience}
- GOAL: {request.business_goal}
- TOPIC: {request.topic_idea}

Generate a JSON object with exactly these keys:
1. "target_keywords": An array of 3-5 high-value, long-tail keywords as strings
2. "content_gap_analysis": A single string describing the biggest missed topic or unique angle

IMPORTANT: Return ONLY the JSON object. No explanation, no markdown formatting, no code fences.

Search snippets:
{search_snippets}
"""
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=gemini_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        raw_gemini_text = response.text.strip()
        print(f"DEBUG: Raw Gemini response: {raw_gemini_text[:200]}...")
        
        raw_gemini_text = raw_gemini_text.replace('```json', '').replace('```', '').strip()
        
        analysis_data = json.loads(raw_gemini_text)
        
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON Parsing Failed: {e}")
        print(f"DEBUG: Raw response was: {raw_gemini_text if 'raw_gemini_text' in locals() else 'No response received'}")
        raise HTTPException(status_code=500, detail={"error": "Gemini analysis failed to return valid JSON.", "raw_llm_response": raw_gemini_text if 'raw_gemini_text' in locals() else "No response"})
    except Exception as e:
        print(f"DEBUG: Gemini analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {e}")
    
    return ResearchData(
        target_keywords=analysis_data.get('target_keywords', []),
        top_competitor_urls=top_urls,
        content_gap_analysis=analysis_data.get('content_gap_analysis', 'No gap defined.')
    )


def calculate_word_count_for_duration(seconds: int) -> int:
    """
    Calculate approximate word count for speech duration.
    Average speaking pace: 150 words per minute (2.5 words per second)
    For natural, clear speech: 130-140 WPM (2.2-2.3 words per second)
    """
    words_per_second = 2.3
    return int(seconds * words_per_second)


def strategic_synthesis_and_blueprint(request: ContentRequest, research_data: ResearchData) -> CreativeBlueprint:
    """Step 2.3: Uses Groq/Gemini for rapid synthesis and structured blueprint generation, outputting final prompts."""
    
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized.")
    
    print("\n--- DEBUG: STEP 2.3 STARTING (Synthesis) ---")

    video_word_count = None
    video_duration_info = ""
    if request.video_duration_seconds:
        video_word_count = calculate_word_count_for_duration(request.video_duration_seconds)
        video_word_count = video_word_count -10
        video_duration_info = f"\nVideo Duration: {request.video_duration_seconds} seconds (Target: {video_word_count} words for natural speech)"

    synthesis_prompt = f"""
You are an elite Creative Strategist at a world-class AI Content Studio. Your mission is to transform the strategic intelligence provided below into a Unified Creative Blueprint. The output must be a single JSON object that contains three fields named blog_text_prompt, image_prompt, and video_prompt. Each field must contain plain English text only. Do not use backticks, markdown formatting, headings, bullets, numbered lists, or nested JSON. The text inside each field must be a single continuous paragraph. Do not escape quotes or newlines - write normal text. The content inside each field must be self-contained so that another agent, receiving that field alone, can generate exceptional output.

Strategic Intelligence Provided:
Core Topic: {request.topic_idea}
Unique Angle: {research_data.content_gap_analysis}
Brand Tone: {request.brand_tone}
Brand Color: {request.brand_color_hex}
Business Objective: {request.business_goal}
Target Keywords: {research_data.target_keywords}
Format Specification: {request.primary_format}, Length: {request.word_count or request.video_duration_seconds}{video_duration_info}

Your Task:
Create three standalone prompt texts: one for generating a long-form blog article, one for generating a hero image, and one for generating video speech script.

CRITICAL REQUIREMENTS FOR video_prompt:
The video_prompt field must generate a SPEECH SCRIPT ONLY - pure spoken words that will be converted to audio using text-to-speech technology (ElevenLabs). This is NOT a description or instruction, it is the ACTUAL WORDS that will be spoken aloud.

For video_prompt, you must:
- Generate approximately {video_word_count if video_word_count else '100-150'} words of natural, conversational speech text
- Write as if you are the narrator speaking directly to the viewer
- Use short, punchy sentences that sound natural when spoken aloud
- Include natural pauses (use periods and commas thoughtfully)
- Avoid complex words or jargon that are hard to pronounce
- Make it engaging, dynamic, and hook the listener immediately
- Match the {request.brand_tone} tone perfectly
- DO NOT include ANY stage directions, descriptions, or meta-instructions
- DO NOT include phrases like "Say this:" or "The script is:" - just write the pure speech text
- THIS IS CRITICAL: The word count MUST match {video_word_count if video_word_count else '100-150'} words for the audio to fit the {request.video_duration_seconds if request.video_duration_seconds else 30} second duration
- Also Focus on the Product & the core idea behind it. It should not be like you are just generating the script that's totally off from the product and not showing anything about the actual product.Keep it like your are advertising the product and it's relevancy.

Example of CORRECT video_prompt output:
"Hey there. Ever wondered why your morning routine feels so chaotic? Here's the truth. You're not broken. Your system is. In the next thirty seconds, I'm going to show you three micro-habits that will transform your entire day. No fluff. No BS. Just results. Ready? Let's dive in."

Example of WRONG video_prompt output:
"Create a video script about morning routines. The narrator should sound energetic and discuss three habits. Include a hook at the beginning."

The image_prompt should describe a hero image with variation, showing the actual product beautifully with photorealistic quality.Focus should be on product . The colour, brand tone should also be focussed.
The image should shows the brand product in such a way that customer get's attracted toward it .Must have a focus on Brand Color: {request.brand_color_hex} for either core product or background .In prompt ,mention the colour by convert the hex code .
Note :- Coluour must be there in the image generation prompt .It's mandatory .Convert hexcode into correct exact colour .

The blog length should be within {request.item_count} mentioned in the blog prompt.Be strict about the word limit of the blog prompt   .
Return ONLY valid JSON with this exact structure (no code fences, no markdown):
{{"blog_text_prompt": "your blog prompt text here", "image_prompt": "your image prompt text here", "video_prompt": "your direct speech script here"}}
"""
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8
            )
        )
        
        blueprint_text = response.text.strip()
        print("DEBUG: Raw Blueprint Received.")
        print(blueprint_text[:500])
        
        blueprint_text = blueprint_text.replace('```json', '').replace('```', '').strip()
        
        blueprint_data = json.loads(blueprint_text)  
        
        for key in ['blog_text_prompt', 'image_prompt', 'video_prompt']:
            if key not in blueprint_data:
                blueprint_data[key] = f"Generate content for {key}"
            elif not isinstance(blueprint_data[key], str):
                blueprint_data[key] = str(blueprint_data[key])

        if request.video_duration_seconds and 'video_prompt' in blueprint_data:
            actual_word_count = len(blueprint_data['video_prompt'].split())
            expected_word_count = calculate_word_count_for_duration(request.video_duration_seconds)
            
            # Check if word count is significantly off (more than 20% difference)
            if actual_word_count < expected_word_count * 0.8:
                print(f"⚠️  WARNING: Script too short ({actual_word_count} words, need {expected_word_count})")
                print("🔄 Regenerating script with stricter instructions...")
                
                # Retry with more explicit instructions
                retry_prompt = f"""The previous script was too short. Generate a speech script with EXACTLY {expected_word_count} words (currently you have {actual_word_count} words, you need {expected_word_count - actual_word_count} MORE words).

Topic: {request.topic_idea}
Duration: {request.video_duration_seconds} seconds
Required word count: {expected_word_count} words

Write a natural, conversational speech script that is EXACTLY {expected_word_count} words. Add more details, examples, and enthusiasm to reach the word count. This is the ACTUAL speech that will be spoken, not a description.

Return ONLY the script text, no JSON, no formatting."""
                
                retry_response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=retry_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.9
                    )
                )
                
                blueprint_data['video_prompt'] = retry_response.text.strip()
                actual_word_count = len(blueprint_data['video_prompt'].split())
                print(f"✅ Regenerated script: {actual_word_count} words")
        return CreativeBlueprint(**blueprint_data)

    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON Parsing failed during synthesis: {e}")
        print(f"DEBUG: Problematic JSON: {blueprint_text[:500]}")
        raise HTTPException(status_code=500, detail={"error": f"Synthesis failed - invalid JSON: {e}", "raw_llm_response": blueprint_text[:1000]})
    except Exception as e:
        print(f"DEBUG: Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail={"error": f"Synthesis failed: {e}", "raw_llm_response": blueprint_text if 'blueprint_text' in locals() else "No response"})


def vary_image_prompt(base_prompt: str, index: int) -> str:
    """
    Creates distinct, high-quality prompts by combining a unique camera angle, 
    a unique lighting style, and a unique abstract background from fixed lists.
    """
    
    # 🧪 EXPANDED: List of pre-defined, abstract, varied backgrounds (18 items).
     

    # 🧪 EXPANDED: List of composition variations (18 items).
    composition_variations = [
        "low angle shot, wide cinematic framing",
        "extreme close-up, shallow depth of field",
        "overhead flat lay, minimalist composition",
        "medium shot, rule of thirds, dynamic angle",
        "high-angle perspective, clean floor",
        "isometric 3D render view, technical drawing style",
        "Dutch angle (canted frame), dynamic energy",
        "close-up macro shot, focus on texture",
        "long shot, emphasizing product scale",
        "panoramic framing, ultra-wide view",
        "focus stacking, edge-to-edge sharpness",
        "silhouette against a bright light source",
        "centered composition, strong symmetry",
        "diagonal composition, leading lines",
        "stack composition, product repeated vertically",
        "split composition, two balanced halves",
        "wide-angle lens distortion, dynamic foreground",
        "shallow depth of field, sharp foreground subject",
    ]
    
    # 🧪 EXPANDED: List of style and lighting variations (18 items).
    style_variations = [
        "dramatic rim lighting, rich contrast",
        "soft daylight, warm pastels",
         
         
        "moody chiaroscuro lighting",
        "bright studio photography, clean white balance",
        "backlit, ethereal glow",
        "high key lighting, pure white look",
        "low key lighting, dark and dramatic",
        "subsurface scattering, glowing effect",
        "volumetric lighting, dust particles visible",
        "soft box lighting, commercial quality",
        "hard direct light, sharp shadows",
        "cross-hatch lighting, security camera aesthetic",
        "magenta and cyan split lighting",
        "iridescent finish, changing colors",
        "diffused natural light, soft shadows"
    ]
    
    # Use index to cycle through all three lists independently
     
    comp_tag = composition_variations[index % len(composition_variations)]
    style_tag = style_variations[index % len(style_variations)]
    
    # Clean up base prompt from LLM
    base_prompt = base_prompt.strip().rstrip(',').rstrip('.')
    
    # Move quality and style modifiers to the END
    quality_suffix = "photorealistic, 8K, ultra-detailed, cinematic" 
    
    # ⭐ CONSTRUCT THE NEW PROMPT: 
    # 1. LLM Core (Subject, Color, Tone)
    # 2. Hard-Coded Background (Guaranteed variety)
    # 3. Hard-Coded Style/Composition
    # 4. Hard-Coded Quality
    new_prompt = f"{base_prompt},{comp_tag}, {style_tag}, {quality_suffix}"
    
    # FINAL CHECK: ENSURE prompt is under the recommended 300 chars
    if len(new_prompt) > 300:
        new_prompt = new_prompt[:300].rsplit(',', 1)[0]
    
    return new_prompt

# --- PHASE 3 GENERATION FUNCTIONS ---

def generate_blog_text(prompt: str) -> str:
    """Generates a blog post using Groq's LLM."""
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized for text generation.")
    
    print("\n--- DEBUG: GENERATING BLOG TEXT (Groq) ---")
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a world-class content writer."},
                {"role": "user", "content": prompt + "Does not include meta description & keywords info ,just only provides the blog & nothing else ."+"Word length must be within :{request.}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"DEBUG: Blog text generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Blog text generation failed: {e}")


def generate_image(prompt: str, image_filename: str = "output_image.png", seed: Optional[int] = None) -> str:
    """
    Generates an image using Fireworks AI FLUX.1-schnell-fp8 model.
    Cost: $0.0014 per image (4 steps × $0.00035)
    """
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    if not FIREWORKS_API_KEY:
        raise HTTPException(status_code=500, detail="Fireworks API key not available for image generation.")

    # Fireworks endpoint
    API_URL = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "image/jpeg",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    }
    print("The prompt for the audio Scripting :",prompt)
    # 🔥 FIX: Truncate prompt to ~300 characters (FLUX limit is around 77 tokens)
    MAX_PROMPT_LENGTH = 300
    if len(prompt) > MAX_PROMPT_LENGTH:
        print(f"⚠️  Prompt too long ({len(prompt)} chars), truncating to {MAX_PROMPT_LENGTH}")
        prompt = prompt[:MAX_PROMPT_LENGTH].rsplit(' ', 1)[0]  # Cut at last word boundary
    
    payload = {"prompt": prompt}
    if seed is not None:
        payload["seed"] = seed

    print(f"\n--- DEBUG: GENERATING IMAGE: {image_filename} (Fireworks AI) ---")
    print(f"📝 Prompt ({len(prompt)} chars): {prompt[:100]}...")
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            file_path = f"generated_content/{image_filename}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            file_size = len(response.content) / 1024
            print(f"✅ Image generated → {file_path} ({file_size:.1f} KB)")
            return file_path
        else:
            print(f"❌ Error generating image: {response.status_code}, {response.text}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {response.status_code}, {response.text}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=500, detail="Image generation timed out.")
    except Exception as e:
        print(f"DEBUG: Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitize text to create safe folder/file names.
    """
    safe_text = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in text[:max_length])
    safe_text = safe_text.strip().replace(' ', '_')
    # Remove multiple underscores
    while '__' in safe_text:
        safe_text = safe_text.replace('__', '_')
    return safe_text.strip('_')


# --- 4. FASTAPI ENDPOINT & PIPELINE ---

app = FastAPI(title="AI Content Agent Backend")

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

os.makedirs("generated_content", exist_ok=True)

origins = [
    "http://localhost:8501",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/generated_content", StaticFiles(directory="generated_content"), name="generated_content")

# Add this to your backend main.py, update the submit_content_brief endpoint:

@app.post("/submit-content-brief")
async def submit_content_brief(request: ContentRequest):
    if not all([groq_client, gemini_client, tavily_client, HF_TOKEN]):
        raise HTTPException(status_code=500, detail="API clients or HF_TOKEN failed to initialize. Check environment variables.")

    print(f"\n--- PHASE 1 COMPLETE: Starting Research for {request.topic_idea} ---")
    
    # --- PHASE 2: Research & Synthesis ---
    research_result = research_and_gap_analysis(request)
    synthesis_result = strategic_synthesis_and_blueprint(request, research_result)
    print("**"*80)
    print(synthesis_result)
    
    # --- PHASE 3: Parallel Generation ---
    generated_blog_content = None
    generated_image_paths = []
    video_path = None
    video_thumbnail_path = None
    video_folder = None

    if request.primary_format == 'Blog Post':
        generated_blog_content = generate_blog_text(synthesis_result.blog_text_prompt)
        
    elif request.primary_format == 'Image':
        num_images = request.item_count or 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = sanitize_filename(request.topic_idea)
        
        images_folder = f"generated_content/images/{timestamp}_{safe_topic}"
        os.makedirs(images_folder, exist_ok=True)
        
        for i in range(num_images):
            image_filename = f"{images_folder}/image_{i+1:02d}.png"
            unique_prompt = vary_image_prompt(synthesis_result.image_prompt, i)
            print(f"DEBUG: Generating image with unique prompt (Index {i+1}): {unique_prompt[:100]}...")
            
            random_seed = random.randint(1, 10000000) 
            generated_image_paths.append(generate_image(
                unique_prompt, 
                image_filename,
                seed=random_seed 
            ))
            time.sleep(0.01)
            
    elif request.primary_format == 'Video':
        print("\n--- DEBUG: GENERATING VIDEO ---")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = sanitize_filename(request.topic_idea)
        
        video_folder = f"generated_content/videos/{timestamp}_{safe_topic}"
        frames_folder = f"{video_folder}/frames"
        os.makedirs(frames_folder, exist_ok=True)
        
        video_thumbnail_path = f"{video_folder}/thumbnail.png"
        video_output_filename = f"{video_folder}/{safe_topic}_final.mp4"
        audio_filename = f"{video_folder}/{safe_topic}_audio.mp3"
        
        print(f"DEBUG: Generating thumbnail for video...")
        generate_image(synthesis_result.image_prompt, video_thumbnail_path)
        print(f"✅ Thumbnail saved: {video_thumbnail_path}")
        
        # ⭐ NEW: Determine logo settings based on user tier
        should_add_logo = (request.user_tier == 'free')
        logo_file_path = "backend/assets/logo.png"
        
        # Validate logo exists
        if should_add_logo and not os.path.exists(logo_file_path):
            print(f"⚠️  WARNING: Logo requested but file not found at {logo_file_path}")
            print(f"⚠️  Continuing without logo...")
            should_add_logo = False
        
        if should_add_logo:
            print(f"🎨 User tier: FREE - Adding logo watermark")
        else:
            print(f"⭐ User tier: PREMIUM - No logo watermark")
        
        try:
            video_script = synthesis_result.video_prompt
            video_duration = request.video_duration_seconds or 30
            
            print(f"DEBUG: Video script: {video_script[:200]}...")
            print(f"DEBUG: Video duration: {video_duration} seconds")
            
            video_path = create_video_from_text(
                text=video_script,
                duration=video_duration,
                output_video=video_output_filename,
                audio_output=audio_filename,
                frames_folder=frames_folder,
                thumbnail_path=video_thumbnail_path,
                seed=random.randint(1, 10000000),
                add_subtitles=True,
                subtitle_style="netflix",
                add_logo=should_add_logo,  # ⭐ NEW
                logo_path=logo_file_path if should_add_logo else None,  # ⭐ NEW
                logo_position="bottom-right",  # ⭐ NEW
                logo_size=150  # ⭐ NEW (you can make this configurable too)
            )
            
            print(f"✅ Video generated successfully: {video_path}")
            if should_add_logo:
                print(f"🎨 Logo watermark applied (free tier)")
            
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")
    
    elif request.primary_format == 'Campaign':
        print("\n--- DEBUG: GENERATING FULL CAMPAIGN ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = sanitize_filename(request.topic_idea)
        
        # 1. Generate Blog
        print("📝 Generating blog post...")
        generated_blog_content = generate_blog_text(synthesis_result.blog_text_prompt)
        
        # 2. Generate Images
        num_images = request.item_count or 2
        images_folder = f"generated_content/images/{timestamp}_{safe_topic}_campaign"
        os.makedirs(images_folder, exist_ok=True)
        
        print(f"🖼️  Generating {num_images} images...")
        for i in range(num_images):
            image_filename = f"{images_folder}/image_{i+1:02d}.png"
            unique_prompt = vary_image_prompt(synthesis_result.image_prompt, i)
            print(f"DEBUG: Generating campaign image {i+1} with prompt: {unique_prompt[:100]}...")
            
            random_seed = random.randint(1, 10000000)
            generated_image_paths.append(generate_image(
                unique_prompt,
                image_filename,
                seed=random_seed
            ))
            time.sleep(0.01)
        
        # 3. Generate Video
        print("🎬 Generating video...")
        video_folder = f"generated_content/videos/{timestamp}_{safe_topic}_campaign"
        frames_folder = f"{video_folder}/frames"
        os.makedirs(frames_folder, exist_ok=True)
        
        video_thumbnail_path = f"{video_folder}/thumbnail.png"
        video_output_filename = f"{video_folder}/{safe_topic}_final.mp4"
        audio_filename = f"{video_folder}/{safe_topic}_audio.mp3"
        
        generate_image(synthesis_result.image_prompt, video_thumbnail_path)
        
        # ⭐ NEW: Logo settings for campaign video
        should_add_logo = (request.user_tier == 'free')
        logo_file_path = "backend/assets/logo.png"
        
        if should_add_logo and not os.path.exists(logo_file_path):
            print(f"⚠️  WARNING: Logo file not found, continuing without logo")
            should_add_logo = False
        
        if should_add_logo:
            print(f"🎨 Campaign video: Adding logo (free tier)")
        else:
            print(f"⭐ Campaign video: No logo (premium tier)")
        
        try:
            video_script = synthesis_result.video_prompt
            video_duration = request.video_duration_seconds or 30
            
            video_path = create_video_from_text(
                text=video_script,
                duration=video_duration,
                output_video=video_output_filename,
                audio_output=audio_filename,
                frames_folder=frames_folder,
                thumbnail_path=video_thumbnail_path,
                seed=random.randint(1, 10000000),
                add_subtitles=True,
                subtitle_style="netflix",
                add_logo=should_add_logo,  # ⭐ NEW
                logo_path=logo_file_path if should_add_logo else None,  # ⭐ NEW
                logo_position="bottom-right",  # ⭐ NEW
                logo_size=150  # ⭐ NEW
            )
            
            print(f"✅ Campaign generation complete!")
            if should_add_logo:
                print(f"🎨 Video includes logo watermark (free tier)")
            
        except Exception as e:
            print(f"❌ Campaign video generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Campaign video generation failed: {str(e)}")

    final_blueprint = {
        "status": "success",
        "message": "Phase 2 & 3 Complete. Content Generated.",
        "inputs": request.model_dump(exclude_none=True),
        "research_data": research_result.model_dump(),
        "creative_blueprint": synthesis_result.model_dump(),
        "generated_content": {
            "blog_post": generated_blog_content,
            "image_file_paths": generated_image_paths,
            "video_file_path": video_path,
            "video_thumbnail_path": video_thumbnail_path,
            "video_folder": video_folder
        }
    }

    print("--- PHASE 2 & 3 COMPLETE ---")
    return final_blueprint


@app.get("/video/{filename}")
async def get_video(filename: str):
    """Serve generated video files"""
    video_path = f"generated_content/{filename}"
    if os.path.exists(video_path):
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=filename
        )
    raise HTTPException(status_code=404, detail="Video not found")