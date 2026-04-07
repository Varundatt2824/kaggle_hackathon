"""
Chat component — follow-up conversation about the report.

Uses Streamlit's chat UI elements (st.chat_message, st.chat_input)
which provide a familiar messaging interface out of the box.
"""

import requests
import streamlit as st

from agent.config import settings


def render_chat_section():
    """Render the follow-up chat interface."""
    result = st.session_state.analysis_result

    st.header("Ask Follow-up Questions")
    st.caption("Ask anything about your report. Responses are grounded in your actual results.")

    # --- Display chat history ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input ---
    user_input = st.chat_input("Ask a question about your report...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = _call_chat_api(user_input, result)

            if response:
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                st.error("Failed to get a response. Please try again.")


def _call_chat_api(message: str, analysis_result: dict) -> str | None:
    """Call the FastAPI /chat endpoint."""
    api_url = f"{settings.api_base_url}/api/v1/chat"

    try:
        response = requests.post(
            api_url,
            json={
                "message": message,
                "report_type": analysis_result["report_type"],
                "findings": analysis_result["findings"],
                "explanation": analysis_result["explanation"],
                "output_language": "English",
                "chat_history": st.session_state.chat_history[:-1],
            },
            timeout=300,
        )

        if response.status_code == 200:
            return response.json()["response"]
        return None

    except (requests.ConnectionError, requests.Timeout):
        return None
