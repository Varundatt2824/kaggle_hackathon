"""
MedSimplify — Streamlit Frontend.

This is the main UI file. Streamlit works top-to-bottom: every time
a user interacts with a widget, the entire script re-runs. To keep
state between re-runs (like analysis results), we use st.session_state.

Run with: uv run streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add project root to Python path so package imports work
# when Streamlit runs this file directly (not as a module).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.components.upload import render_upload_section
from app.components.results import render_results_section
from app.components.chat import render_chat_section

# --- Page config (must be first Streamlit command) ---
st.set_page_config(
    page_title="MedSimplify",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize session state ---
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False
if "translated_explanation" not in st.session_state:
    st.session_state.translated_explanation = None
if "translated_questions" not in st.session_state:
    st.session_state.translated_questions = []
if "translated_language" not in st.session_state:
    st.session_state.translated_language = ""


def main():
    # --- Header ---
    st.title("MedSimplify")
    st.markdown("*Understand your medical reports in plain language*")

    # --- Safety disclaimer banner ---
    st.warning(
        "**Disclaimer:** MedSimplify is an educational tool and is NOT a substitute "
        "for professional medical advice, diagnosis, or treatment. Always consult "
        "your healthcare provider about your results.",
        icon="⚠️",
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("About")
        st.markdown("**How it works:**")
        st.markdown(
            "1. Upload your medical report\n"
            "2. Get a plain-language explanation in English\n"
            "3. Translate into any language you need\n"
            "4. Ask follow-up questions"
        )

        st.divider()
        st.caption(
            "Powered by Gemma 4 via Ollama. "
            "Your data stays on your device."
        )

    # --- Main content ---

    # Step 1: Upload section (always analyzes in English)
    render_upload_section()

    # Step 2: Results + translation (shown after analysis)
    if st.session_state.analysis_result:
        render_results_section()

        st.divider()

        # Step 3: Chat follow-up (always in English — user can ask in any language though)
        render_chat_section()


if __name__ == "__main__":
    main()
