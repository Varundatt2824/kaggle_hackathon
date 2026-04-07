"""
Upload component — handles file upload and triggers analysis.

Why a separate component file?
Keeps main.py clean and focused on layout. Each component manages
its own UI logic. This is similar to React components but in Python.
"""

import base64

import requests
import streamlit as st

from agent.config import settings


def render_upload_section():
    """Render the file upload UI and handle analysis trigger."""

    st.header("Upload Your Report")

    # --- Input method tabs ---
    tab_upload, tab_paste = st.tabs(["Upload File", "Paste Text"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a medical report (PDF or image)",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Supported formats: PDF, PNG, JPG",
        )

        if uploaded_file and not st.session_state.is_analyzing:
            if st.button("Analyze Report", type="primary", key="analyze_file"):
                _run_analysis_file(uploaded_file)

    with tab_paste:
        pasted_text = st.text_area(
            "Paste your medical report text here",
            height=300,
            placeholder="Paste the contents of your lab report or radiology summary...",
        )

        if pasted_text and not st.session_state.is_analyzing:
            if st.button("Analyze Report", type="primary", key="analyze_text"):
                _run_analysis_text(pasted_text)


def _run_analysis_file(uploaded_file):
    """Process an uploaded file through the analysis API."""
    st.session_state.is_analyzing = True
    st.session_state.analysis_result = None
    st.session_state.chat_history = []
    st.session_state.translated_explanation = None

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".pdf"):
        input_type = "pdf"
    else:
        input_type = "image"

    raw_input = base64.b64encode(uploaded_file.read()).decode("utf-8")
    _call_analyze_api(raw_input, input_type, uploaded_file.name)


def _run_analysis_text(text: str):
    """Process pasted text through the analysis API."""
    st.session_state.is_analyzing = True
    st.session_state.analysis_result = None
    st.session_state.chat_history = []
    st.session_state.translated_explanation = None

    _call_analyze_api(text, "text", "pasted_report.txt")


def _call_analyze_api(raw_input: str, input_type: str, file_name: str):
    """Call the FastAPI /analyze endpoint and store the result.

    Always analyzes in English for maximum accuracy. Translation
    happens as a separate step after the analysis.
    """
    api_url = f"{settings.api_base_url}/api/v1/analyze"

    with st.spinner("Analyzing your report... This may take a minute."):
        try:
            response = requests.post(
                api_url,
                json={
                    "raw_input": raw_input,
                    "input_type": input_type,
                    "file_name": file_name,
                    "output_language": "English",
                },
                timeout=300,
            )

            if response.status_code == 200:
                st.session_state.analysis_result = response.json()
                st.session_state.is_analyzing = False
                st.rerun()
            else:
                try:
                    detail = response.json().get("detail", "Unknown error")
                except Exception:
                    detail = response.text or f"HTTP {response.status_code}"
                st.error(f"Analysis failed: {detail}")
                st.session_state.is_analyzing = False

        except requests.ConnectionError:
            st.error(
                "Cannot connect to the API server. "
                "Make sure the backend is running: `make api`"
            )
            st.session_state.is_analyzing = False
        except requests.Timeout:
            st.error("Analysis timed out. The report may be too large or the model is busy.")
            st.session_state.is_analyzing = False
