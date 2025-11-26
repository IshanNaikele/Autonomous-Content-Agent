import streamlit as st
import requests
import os
from PIL import Image

# --- CONFIGURATION ---
FASTAPI_URL = "http://localhost:8000"
SUBMIT_ENDPOINT = f"{FASTAPI_URL}/submit-content-brief"

# --- PAGE SETUP ---
st.set_page_config(page_title="🤖 AI Content Agent", layout="wide")
st.title("🤖 AI Content Agent")
st.markdown("### Strategic Content Generation Platform")

# Initialize session state
if 'phase_complete' not in st.session_state:
    st.session_state.phase_complete = False
    st.session_state.blueprint = None

# --- SUBMISSION FORM ---
if not st.session_state.phase_complete:
    with st.form(key='content_brief_form'):
        
        # Core Topic
        st.header("1. Core Topic")
        topic_idea = st.text_area(
            "💡 What content do you want to create?",
            placeholder="e.g., Top 5 ways AI is transforming small business operations",
            height=100
        )
        
        # Content Type Selection
        st.header("2. Content Type")
        content_type = st.radio(
            "Select content type:",
            ['Blog', 'Image', 'Video', 'Campaign'],
            horizontal=True
        )
        
        # Dynamic fields based on content type
        st.header("3. Content Specifications")
        
        item_count = None
        word_count = None
        video_duration_seconds = None
        
        if content_type == 'Blog':
            word_count = st.number_input(
                "📝 Word Count",
                min_value=100,
                max_value=5000,
                value=1500,
                step=100
            )
            item_count = 1
            primary_format = 'Blog Post'
            
        elif content_type == 'Image':
            primary_format = 'Image'
            item_count = st.slider(
                "🖼️ Number of Images to Generate",
                min_value=1,
                max_value=3,
                value=1,
                step=1,
                help="Select how many variations of images you want to generate"
            )
            st.success(f"✨ Will generate **{item_count}** image{'s' if item_count > 1 else ''} based on your topic")
            st.caption(f"💡 Tip: Select {item_count} {'image' if item_count == 1 else 'images'} will take approximately {item_count * 30} seconds to generate")
            
        elif content_type == 'Video':
            video_duration_seconds = st.slider(
                "⏱️ Video Duration (seconds)",
                min_value=5,
                max_value=30,
                value=30,
                step=5
            )
            item_count = 1
            primary_format = 'Video'
            
        elif content_type == 'Campaign':
            st.info("📦 Campaign will generate Blog + Image + Video")
            word_count = st.number_input(
                "📝 Blog Word Count",
                min_value=100,
                max_value=5000,
                value=1500,
                step=100
            )
            item_count = st.slider(
                "🖼️ Number of Images",
                min_value=1,
                max_value=3,
                value=2,
                help="Select how many variations of images you want for the campaign"
            )
            st.info(f"✨ Will generate **{item_count}** image{'s' if item_count > 1 else ''}")
            video_duration_seconds = st.slider(
                "⏱️ Video Duration (seconds)",
                min_value=5,
                max_value=30,
                value=30,
                step=5
            )
            primary_format = 'Blog Post'  # Default for campaign
        
        st.markdown("---")
        
        # Target & Goal
        st.header("4. Target Audience & Goal")
        col1, col2 = st.columns(2)
        with col1:
            target_audience = st.text_area(
                "🎯 Target Audience",
                placeholder="e.g., Mid-level marketing managers in tech startups",
                height=80
            )
        with col2:
            business_goal = st.selectbox(
                "📈 Business Goal",
                ['Awareness', 'Lead Generation', 'Conversion', 'Engagement']
            )
        
        st.markdown("---")
        
        # Brand Style
        st.header("5. Brand Style")
        col3, col4 = st.columns(2)
        with col3:
            brand_tone = st.selectbox(
                "🗣️ Brand Tone",
                ['Expert & Formal', 'Friendly & Witty', 'Casual & Rebellious']
            )
        with col4:
            brand_color_hex = st.color_picker("🎨 Brand Color", '#007bff')
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Generate Content", use_container_width=True)
    
    # --- SUBMISSION LOGIC ---
    if submitted:
        if not topic_idea or not target_audience:
            st.error("⚠️ Please fill in Topic and Target Audience fields.")
            st.stop()
        
        # Build payload
        payload = {
            "topic_idea": topic_idea,
            "primary_format": primary_format,
            "target_audience": target_audience,
            "business_goal": business_goal,
            "brand_tone": brand_tone,
            "brand_color_hex": brand_color_hex
        }
        
        # Add conditional fields
        if item_count is not None:
            payload["item_count"] = item_count
        if word_count is not None:
            payload["word_count"] = word_count
        if video_duration_seconds is not None:
            payload["video_duration_seconds"] = video_duration_seconds
        
        # Make API call
        with st.spinner(f"🔄 Generating your content... This may take up to {60 + (item_count or 1) * 30} seconds."):
            try:
                response = requests.post(SUBMIT_ENDPOINT, json=payload, timeout=150)
                
                if response.status_code == 200:
                    st.session_state.blueprint = response.json()
                    st.session_state.phase_complete = True
                    st.success("✅ Content generated successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Error: Status {response.status_code}")
                    try:
                        st.json(response.json())
                    except:
                        st.code(response.text)
                        
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to backend at {FASTAPI_URL}. Is it running?")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The generation took too long.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# --- RESULTS DISPLAY ---
if st.session_state.phase_complete and st.session_state.blueprint:
    blueprint = st.session_state.blueprint
    generated_content = blueprint.get('generated_content', {})
    research_data = blueprint.get('research_data', {})
    creative_blueprint = blueprint.get('creative_blueprint', {})
    
    st.header("✨ Generated Content")
    
    # Display Multiple Images
    if generated_content.get('image_file_paths'):
        image_paths = generated_content['image_file_paths']
        
        if len(image_paths) == 1:
            st.subheader("🖼️ Generated Image")
            image_path = image_paths[0]
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image: {e}")
            else:
                st.warning(f"Image file not found: {image_path}")
        else:
            st.subheader(f"🖼️ Generated Images ({len(image_paths)} variations)")
            # Create columns for multiple images
            cols = st.columns(min(len(image_paths), 3))
            for idx, image_path in enumerate(image_paths):
                with cols[idx % 3]:
                    if os.path.exists(image_path):
                        try:
                            img = Image.open(image_path)
                            st.image(img, caption=f"Image {idx + 1}", use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not load image {idx + 1}: {e}")
                    else:
                        st.warning(f"Image {idx + 1} not found")
    
    # Display Blog Post
    if generated_content.get('blog_post'):
        st.subheader("📝 Generated Blog Post")
        with st.expander("View Full Blog Post", expanded=True):
            st.markdown(generated_content['blog_post'])
    
    # Display Video Script (if implemented)
    if generated_content.get('video_script'):
        st.subheader("🎬 Video Script")
        with st.expander("View Video Script"):
            st.markdown(generated_content['video_script'])
    
    st.divider()
    
    # Research & Blueprint Details
    with st.expander("📊 Research & Strategy Details"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Content Gap Analysis:**")
            st.info(research_data.get('content_gap_analysis', 'N/A'))
            
            st.markdown("**Target Keywords:**")
            keywords = research_data.get('target_keywords', [])
            for kw in keywords:
                st.caption(f"• {kw}")
        
        with col_b:
            st.markdown("**Top Competitor URLs:**")
            urls = research_data.get('top_competitor_urls', [])
            for url in urls:
                st.caption(f"• {url}")
    
    with st.expander("🎨 Generation Prompts Used"):
        if creative_blueprint.get('blog_text_prompt'):
            st.markdown("**Blog Prompt:**")
            st.code(creative_blueprint['blog_text_prompt'], language="text")
        
        if creative_blueprint.get('image_prompt'):
            st.markdown("**Image Prompt:**")
            st.code(creative_blueprint['image_prompt'], language="text")
        
        if creative_blueprint.get('video_prompt'):
            st.markdown("**Video Prompt:**")
            st.code(creative_blueprint['video_prompt'], language="text")
    
    st.divider()
    
    # Reset button
    if st.button("🔄 Create New Content", use_container_width=True):
        st.session_state.phase_complete = False
        st.session_state.blueprint = None
        st.rerun()