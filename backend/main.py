# backend/main.py (FINAL CODE - Image Variation and Quality Optimized)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, PositiveInt
from typing import Literal, Optional, List
import os
import json
import time 
import requests
import random # For random seed and prompt variation selection

# --- API CLIENTS ---
from groq import Groq
from google import genai
from tavily import TavilyClient
from google.genai import types 

# --- 1. Pydantic Schemas for Input and Output ---

class ContentRequest(BaseModel):
    topic_idea: str = Field(..., max_length=500, description="The main subject or idea for the content.")
    primary_format: Literal['Blog Post', 'Image', 'Video'] = Field(..., description="The main deliverable format.")
    
    item_count: Optional[PositiveInt] = Field(None, description="The number of items required (must be >= 1).")
    word_count: Optional[PositiveInt] = Field(None, description="Required word count for Blog Post (must be >= 1).")
    video_duration_seconds: Optional[PositiveInt] = Field(None, description="Required duration for Video format in seconds (must be >= 1).")
    
    target_audience: str = Field(..., description="Persona of the primary reader/viewer.")
    business_goal: Literal['Awareness', 'Lead Generation', 'Conversion', 'Engagement'] = Field(..., description="The goal this content must achieve.")
    brand_tone: Literal['Expert & Formal', 'Friendly & Witty', 'Casual & Rebellious'] = Field(..., description="The required tone of voice.")
    brand_color_hex: Optional[str] = Field(None, description="Primary brand color (e.g., #007bff)")

class ResearchData(BaseModel):
    target_keywords: List[str] = Field(..., description="3-5 inferred high-value keywords.")
    top_competitor_urls: List[str] = Field(..., description="The top 5 URLs found by the search engine.")
    content_gap_analysis: str = Field(..., description="The unique angle/topic the competition missed.")

class CreativeBlueprint(BaseModel):
    blog_text_prompt: str = Field(..., description="The single, complete prompt for the Blog/Text generation agent.")
    image_prompt: str = Field(..., description="The single, complete prompt for the Image generation agent.")
    video_prompt: str = Field(..., description="The single, complete prompt for the Video generation agent.")


# --- 2. API CLIENT INITIALIZATION ---
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not all([GROQ_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY, HF_TOKEN]):
        # Removed the raising of an error here to allow the app to initialize but still print the error
        print("One or more API keys (GROQ_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY, HF_TOKEN) are missing from environment variables.")
        # We will handle the HTTPException inside the endpoint functions
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
        
        # Clean up common formatting issues
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


def strategic_synthesis_and_blueprint(request: ContentRequest, research_data: ResearchData) -> CreativeBlueprint:
    """Step 2.3: Uses Groq/Gemini for rapid synthesis and structured blueprint generation, outputting final prompts."""
    
    if not groq_client: # Using Groq for the main synthesis, but falling back to Gemini if needed/or configured to
        raise HTTPException(status_code=500, detail="Groq client not initialized.")
    
    print("\n--- DEBUG: STEP 2.3 STARTING (Synthesis) ---")

    synthesis_prompt = f"""
You are an elite Creative Strategist at a world-class AI Content Studio. Your mission is to transform the strategic intelligence provided below into a Unified Creative Blueprint. The output must be a single JSON object that contains three fields named blog_text_prompt, image_prompt, and video_prompt. Each field must contain plain English text only. Do not use backticks, markdown formatting, headings, bullets, numbered lists, or nested JSON. The text inside each field must be a single continuous paragraph. Do not escape quotes or newlines - write normal text. The content inside each field must be self-contained so that another agent, receiving that field alone, can generate exceptional output.

Strategic Intelligence Provided:
Core Topic: {request.topic_idea}
Unique Angle: {research_data.content_gap_analysis}
Brand Tone: {request.brand_tone}
Brand Color: {request.brand_color_hex}
Business Objective: {request.business_goal}
Target Keywords: {research_data.target_keywords}
Format Specification: {request.primary_format}, Length: {request.word_count or request.video_duration_seconds}

Your Task:
Create three standalone prompt texts: one for generating a long-form blog article, one for generating a hero image, and one for generating a 30-second vertical video. Write everything as smooth, flowing paragraphs with normal punctuation.
The prompt for image is like the image should have variation & it shows the actual product very nicely .
Return ONLY valid JSON with this exact structure (no code fences, no markdown):
{{"blog_text_prompt": "your blog prompt text here", "image_prompt": "your image prompt text here", "video_prompt": "your video prompt text here"}}
"""
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        blueprint_text = response.text.strip()
        print("DEBUG: Raw Blog & image Blueprint Received.")
        print(blueprint_text[:500])
        
        # Clean the response
        blueprint_text = blueprint_text.replace('```json', '').replace('```', '').strip()
        
        blueprint_data = json.loads(blueprint_text)  
        
        # Ensure all fields are strings
        for key in ['blog_text_prompt', 'image_prompt', 'video_prompt']:
            if key not in blueprint_data:
                blueprint_data[key] = f"Generate content for {key}"
            elif not isinstance(blueprint_data[key], str):
                blueprint_data[key] = str(blueprint_data[key])

        return CreativeBlueprint(**blueprint_data)

    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON Parsing failed during synthesis: {e}")
        print(f"DEBUG: Problematic JSON: {blueprint_text[:500]}")
        raise HTTPException(status_code=500, detail={"error": f"Synthesis failed - invalid JSON: {e}", "raw_llm_response": blueprint_text[:1000]})
    except Exception as e:
        print(f"DEBUG: Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail={"error": f"Synthesis failed: {e}", "raw_llm_response": blueprint_text if 'blueprint_text' in locals() else "No response"})


# --- IMAGE VARIATION HELPER (Optimized for Quality) ---

def vary_image_prompt(base_prompt: str, index: int) -> str:
    """
    Creates distinct, high-quality prompts by injecting specific camera angles 
    and lighting styles while preserving the original quality modifiers.
    
    This structured approach prevents quality degradation.
    """
    
    # Zone 2: Composition Variations (Camera Angles/Framing)
    composition_variations = [
        "shot from a low angle, dramatic wide-angle cinematic framing",
        "extreme close-up portrait, shallow depth of field (bokeh), subject in sharp focus",
        "overhead flat lay view, minimalist studio background, clean and professional",
        "medium shot, following the rule of thirds composition, leading lines, sense of scale",
        "high-angle perspective, capturing the subject within a larger environmental context",
        "an isometric 3D render, showcasing the product from the top-front view, clean interface",
    ]
    
    # Zone 3: Style and Aesthetic Variations (Lighting/Mood) - Including high-quality stable modifiers
    style_variations = [
        "illuminated by dramatic rim lighting and deep shadows, rich contrast",
        "soft, diffused daylight, warm pastel color palette, light atmosphere",
        "vibrant, high-contrast neon glow, dynamic ambient light, futuristic aesthetic",
        "golden hour sunlight streaming from the right, cinematic warm tones, volumetric light",
        "moody, cinematic lighting, chiaroscuro style, emphasizing texture and materials",
        "bright, high-key studio photography, soft shadows, professional advertisement look",
    ]
    
    # Get indexed variation elements (using modulo for cycling through the list)
    comp_tag = composition_variations[index % len(composition_variations)]
    style_tag = style_variations[index % len(style_variations)]
    
    # Combine the base prompt (Zone 1: Quality/Subject) with the new zones
    base_prompt = base_prompt.strip().rstrip(',').rstrip('.')
    
    # Re-inject high-quality, stable keywords upfront to guide the model's quality output
    quality_prefix = "photorealistic, ultra-detailed, sharp focus, 8K, rendered with Unreal Engine,"
    
    new_prompt = f"{quality_prefix} {base_prompt}, COMPOSITION: {comp_tag}, LIGHTING: {style_tag}. Make it feel like a real image .Focus on the main Product & code idea ."
    
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
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant", # Using the stable model for consistency
            temperature=0.7 # Higher temp for creative writing
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"DEBUG: Blog text generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Blog text generation failed: {e}")

def generate_image(prompt: str, image_filename: str = "output_image.png", seed: Optional[int] = None) -> str:
    """
    Generates an image using Hugging Face's FLUX.1-schnell model.
    Includes random seed in the payload.
    """
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Hugging Face token not available for image generation.")

    API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # --- ADD GENERATION PARAMETERS WITH SEED ---
    generation_parameters = {}
    if seed is not None:
        # Note: The exact key for seed depends on the specific HF model endpoint (TGI vs. standard)
        generation_parameters['seed'] = seed

    payload = {
        "inputs": prompt,
        "parameters": generation_parameters
    }

    print(f"\n--- DEBUG: GENERATING IMAGE: {image_filename} (Hugging Face) ---")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        
        if response.status_code == 200:
            file_path = f"generated_content/{image_filename}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Image generated → {file_path}")
            return file_path
        else:
            print(f"Error generating image: {response.status_code}, {response.text}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {response.status_code}, {response.text}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=500, detail="Image generation timed out. Hugging Face API took too long.")
    except Exception as e:
        print(f"DEBUG: Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")


# --- 4. FASTAPI ENDPOINT & PIPELINE ---

app = FastAPI(title="AI Content Agent Backend")

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Create generated_content directory if it doesn't exist
os.makedirs("generated_content", exist_ok=True)

# Allow all origins for HTML file access
origins = [
    "http://localhost:8501",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "null"  # For file:// protocol
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for images) - MUST be after CORS middleware
app.mount("/generated_content", StaticFiles(directory="generated_content"), name="generated_content")


@app.post("/submit-content-brief")
async def submit_content_brief(request: ContentRequest):
    if not all([groq_client, gemini_client, tavily_client, HF_TOKEN]):
        raise HTTPException(status_code=500, detail="API clients or HF_TOKEN failed to initialize. Check environment variables.")

    print(f"\n--- PHASE 1 COMPLETE: Starting Research for {request.topic_idea} ---")
    
    # --- PHASE 2: Research & Synthesis ---
    research_result = research_and_gap_analysis(request)
    synthesis_result = strategic_synthesis_and_blueprint(request, research_result)
    
    # --- NEW: PHASE 3: Parallel Generation ---
    generated_blog_content = None
    generated_image_paths = []
    # generated_video_content = None

    if request.primary_format == 'Blog Post':
        generated_blog_content = generate_blog_text(synthesis_result.blog_text_prompt)
        
    elif request.primary_format == 'Image':
        # If primary is image, generate the requested number of images
        num_images = request.item_count or 1
        for i in range(num_images):
            # ADD: Create unique timestamp-based filename
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            image_filename = f"image_{timestamp}_{i+1}.png"
            
            # --- QUALITY & VARIATION SOLUTION ---
            unique_prompt = vary_image_prompt(synthesis_result.image_prompt, i)
            print(f"DEBUG: Generating image with unique prompt (Index {i+1}): {unique_prompt[:100]}...")
            
            # Use random seed for additional variation
            random_seed = random.randint(1, 10000000) 
            generated_image_paths.append(generate_image(
                unique_prompt, 
                image_filename,
                seed=random_seed 
            ))
            
            # ADD: Small delay to ensure different timestamps if generating multiple images
            time.sleep(0.01)
            
    elif request.primary_format == 'Video':
    # Generate thumbnail image for the video
        timestamp = int(time.time() * 1000)
        generated_image_paths.append(generate_image(
            synthesis_result.image_prompt, 
            f"video_thumbnail_{timestamp}.png"
        ))

    final_blueprint = {
        "status": "success",
        "message": "Phase 2 & 3 Complete. Content Generated.",
        "inputs": request.model_dump(exclude_none=True),
        "research_data": research_result.model_dump(),
        "creative_blueprint": synthesis_result.model_dump(),
        "generated_content": {
            "blog_post": generated_blog_content,
            "image_file_paths": generated_image_paths,
            # "video_script": generated_video_content
        }
    }

    print("--- PHASE 2 & 3 COMPLETE ---")
    return final_blueprint