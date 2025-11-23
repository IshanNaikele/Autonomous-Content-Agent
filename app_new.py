# app_new.py
import streamlit as st
import requests
import base64
from requests.exceptions import JSONDecodeError

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="Free Text → Video Generator", layout="centered")
st.title("🎬 Free Text → Video Generator")
st.markdown("---")

prompt = st.text_area(
    "Enter your prompt:",
    "A futuristic city with flying cars at night.",
    key="prompt_area"
)

st.info("This free pipeline uses FLUX (image) + SVD (video). Takes ~20–40 sec.")

if st.button("Generate Video"):
    if not prompt:
        st.warning("Enter a prompt!")
    else:
        with st.spinner("Generating video..."):
            res = requests.post(
                f"{BACKEND}/text_to_video_free",
                json={"prompt": prompt}
            )

        if res.status_code != 200:
            st.error(f"Backend error: {res.status_code}")
            st.code(res.text)
        else:
            try:
                data = res.json()
            except JSONDecodeError:
                st.error("Invalid JSON from server.")
                st.code(res.text)
                st.stop()

            if "error" in data:
                st.error(data["error"])
            else:
                video_bytes = base64.b64decode(data["video_base64"])
                st.success("Video generated!")
                st.video(video_bytes)
