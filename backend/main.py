# backend/main.py (MODIFIED to remove platform_constraint and change prompt structure)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, PositiveInt
from typing import Literal, Optional, List
import os
import json
import time 
import requests
# --- API CLIENTS ---
from groq import Groq
from google import genai
from tavily import TavilyClient
from google.genai import types 

# --- 1. Pydantic Schemas for Input and Output ---

# Input Schema (ContentRequest) - REMOVED platform_constraint
class ContentRequest(BaseModel):
    topic_idea: str = Field(..., max_length=500, description="The main subject or idea for the content.")
    primary_format: Literal['Blog Post', 'Image', 'Video'] = Field(..., description="The main deliverable format.")
    
    item_count: Optional[PositiveInt] = Field(None, description="The number of items required (must be >= 1).")
    word_count: Optional[PositiveInt] = Field(None, description="Required word count for Blog Post (must be >= 1).")
    video_duration_seconds: Optional[PositiveInt] = Field(None, description="Required duration for Video format in seconds (must be >= 1).")
    
    # REMOVED: platform_constraint
    target_audience: str = Field(..., description="Persona of the primary reader/viewer.")
    business_goal: Literal['Awareness', 'Lead Generation', 'Conversion', 'Engagement'] = Field(..., description="The goal this content must achieve.")
    brand_tone: Literal['Expert & Formal', 'Friendly & Witty', 'Casual & Rebellious'] = Field(..., description="The required tone of voice.")
    brand_color_hex: Optional[str] = Field(None, description="Primary brand color (e.g., #007bff)")

# Output Schema for Research (Removed platform_best_practices)
class ResearchData(BaseModel):
    target_keywords: List[str] = Field(..., description="3-5 inferred high-value keywords.")
    top_competitor_urls: List[str] = Field(..., description="The top 5 URLs found by the search engine.")
    content_gap_analysis: str = Field(..., description="The unique angle/topic the competition missed.")
    # REMOVED: platform_best_practices

# Output Schema for Synthesis (Blueprint) - Changed to output the final 3 prompts
class CreativeBlueprint(BaseModel):
    blog_text_prompt: str = Field(..., description="The single, complete prompt for the Blog/Text generation agent.")
    image_prompt: str = Field(..., description="The single, complete prompt for the Image generation agent.")
    video_prompt: str = Field(..., description="The single, complete prompt for the Video generation agent.")


# --- 2. API CLIENT INITIALIZATION (Remains the same) ---
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not all([GROQ_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY, HF_TOKEN]):
        raise ValueError("One or more API keys (GROQ_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY, HF_TOKEN) are missing from environment variables.")

    groq_client = Groq(api_key=GROQ_API_KEY)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

except ValueError as e:
    print(f"FATAL ERROR: API clients failed to initialize. {e}")
    groq_client, gemini_client, tavily_client = None, None, None


# --- 3. CORE PHASE 2 FUNCTIONS ---

# Replace the research_and_gap_analysis function with this corrected version

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
    """Step 2.3: Uses Groq for rapid synthesis and structured blueprint generation, outputting final prompts."""
    
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized.")
    
    print("\n--- DEBUG: STEP 2.3 STARTING (Groq Synthesis) ---")

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


# --- NEW: PHASE 3 GENERATION FUNCTIONS ---

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

def generate_image(prompt: str, image_filename: str = "output_image.png") -> str:
    """Generates an image using Hugging Face's FLUX.1-schnell model."""
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Hugging Face token not available for image generation.")

    API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    print("\n--- DEBUG: GENERATING IMAGE (Hugging Face) ---")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90) # Increased timeout
        
        if response.status_code == 200:
            file_path = f"generated_content/{image_filename}"
            # Ensure the directory exists
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

# (We'll skip video generation for now as it's not requested, but the structure would be similar)
# def generate_video_script(prompt: str) -> str:
#    # ... implementation for video script generation ...
# --- 4. FASTAPI ENDPOINT & PIPELINE (Remains the same) ---

app = FastAPI(title="AI Content Agent Backend")

from fastapi.middleware.cors import CORSMiddleware
origins = ["http://localhost:8501"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/submit-content-brief")
async def submit_content_brief(request: ContentRequest):
    if not all([groq_client, gemini_client, tavily_client, HF_TOKEN]): # Check HF_TOKEN here too
         raise HTTPException(status_code=500, detail="API clients or HF_TOKEN failed to initialize. Check environment variables.")

    print(f"\n--- PHASE 1 COMPLETE: Starting Research for {request.topic_idea} ---")
    
    # --- PHASE 2: Research & Synthesis ---
    research_result = research_and_gap_analysis(request)
    synthesis_result = strategic_synthesis_and_blueprint(request, research_result)
    
    # --- NEW: PHASE 3: Parallel Generation ---
    generated_blog_content = None
    generated_image_path = None
    # generated_video_content = None # Not implementing video for now

    if request.primary_format == 'Blog Post':
        generated_blog_content = generate_blog_text(synthesis_result.blog_text_prompt)
        # For a blog post, you might still want a hero image
        # generated_image_path = generate_image(synthesis_result.image_prompt, "blog_hero_image.png")
    elif request.primary_format == 'Image':
        # If primary is image, only generate the image
        generated_image_path = generate_image(synthesis_result.image_prompt, "primary_content_image.png")
    elif request.primary_format == 'Video':
        # If primary is video, generate the video script, and maybe a thumbnail image
        # generated_video_content = generate_video_script(synthesis_result.video_prompt)
        generated_image_path = generate_image(synthesis_result.image_prompt, "video_thumbnail.png") # Generate an image for the video thumbnail

    final_blueprint = {
        "status": "success",
        "message": "Phase 2 & 3 Complete. Content Generated.",
        "inputs": request.model_dump(exclude_none=True),
        "research_data": research_result.model_dump(),
        "creative_blueprint": synthesis_result.model_dump(),
        "generated_content": {
            "blog_post": generated_blog_content,
            "image_file_path": generated_image_path,
            # "video_script": generated_video_content # For future video integration
        }
    }

    print("--- PHASE 2 & 3 COMPLETE ---")
    return final_blueprint