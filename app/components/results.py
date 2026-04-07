"""
Results component — displays the analysis output.

Shows findings, explanation, and suggested follow-up questions
in a structured, easy-to-read layout. Always displays in English first,
then offers a translate button for any language.
"""

import requests
import streamlit as st

from agent.config import settings


def render_results_section():
    """Render the analysis results."""
    result = st.session_state.analysis_result

    st.header("Your Report Explained")

    # --- Report type badge ---
    report_type = result["report_type"].capitalize()
    st.caption(f"Report type: **{report_type}**")

    # --- Main explanation (always English first) ---
    st.markdown(result["explanation"])

    # --- Structured findings (expandable) ---
    with st.expander("View Detailed Findings", expanded=False):
        _render_findings(result["findings"], result["report_type"])

    # --- Follow-up questions ---
    if result["followup_questions"]:
        st.subheader("Questions to Ask Your Doctor")
        for i, question in enumerate(result["followup_questions"], 1):
            st.markdown(f"**{i}.** {question}")

    # --- Translation section ---
    st.divider()
    _render_translation_section(result)


def _render_translation_section(result: dict):
    """Offer translation of the explanation into any language.

    Why translate after showing English?
    Medical terminology is most standardized in English, so the initial
    analysis is most accurate in English. Translating the final explanation
    (not re-analyzing) preserves that accuracy while making it accessible.
    """
    st.subheader("Translate Explanation")

    col1, col2 = st.columns([3, 1])
    with col1:
        target_language = st.text_input(
            "Translate to",
            placeholder="e.g., Hindi, Spanish, Japanese, Swahili...",
            key="translate_language",
        )
    with col2:
        st.write("")  # Spacing to align button with input
        st.write("")
        translate_clicked = st.button("Translate", type="primary", disabled=not target_language)

    # Show translated result if available
    if translate_clicked and target_language:
        _translate_and_show(result, target_language.strip())
    elif "translated_explanation" in st.session_state and st.session_state.translated_explanation:
        st.info(f"**Translated to {st.session_state.translated_language}:**")
        st.markdown(st.session_state.translated_explanation)


def _translate_and_show(result: dict, target_language: str):
    """Call the backend to translate the explanation."""
    api_url = f"{settings.api_base_url}/api/v1/translate"

    with st.spinner(f"Translating to {target_language}..."):
        try:
            response = requests.post(
                api_url,
                json={
                    "text": result["explanation"],
                    "followup_questions": result["followup_questions"],
                    "target_language": target_language,
                },
                timeout=300,
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.translated_explanation = data["translated_text"]
                st.session_state.translated_questions = data["translated_questions"]
                st.session_state.translated_language = target_language

                st.info(f"**Translated to {target_language}:**")
                st.markdown(data["translated_text"])

                if data["translated_questions"]:
                    st.subheader(f"Questions for Your Doctor ({target_language})")
                    for i, q in enumerate(data["translated_questions"], 1):
                        st.markdown(f"**{i}.** {q}")
            else:
                try:
                    detail = response.json().get("detail", "Unknown error")
                except Exception:
                    detail = response.text or f"HTTP {response.status_code}"
                st.error(f"Translation failed: {detail}")

        except requests.ConnectionError:
            st.error("Cannot connect to the API server.")
        except requests.Timeout:
            st.error("Translation timed out.")


def _render_findings(findings: list[dict], report_type: str):
    """Render findings in a structured format."""
    if report_type == "lab":
        _render_lab_findings(findings)
    else:
        _render_radiology_findings(findings)


def _render_lab_findings(findings: list[dict]):
    """Render lab findings as a color-coded table."""
    for finding in findings:
        name = finding.get("name", "Unknown")
        value = finding.get("value", "N/A")
        unit = finding.get("unit", "")
        ref_range = finding.get("ref_range", "N/A")
        status = finding.get("status", "unknown").lower()

        if status == "high":
            icon = "🔴"
        elif status == "low":
            icon = "🔵"
        elif status == "normal":
            icon = "🟢"
        else:
            icon = "⚪"

        st.markdown(
            f"{icon} **{name}**: {value} {unit} "
            f"(Reference: {ref_range}) — *{status}*"
        )


def _render_radiology_findings(findings: list[dict]):
    """Render radiology findings as a list."""
    for finding in findings:
        text = finding.get("finding", "Unknown finding")
        region = finding.get("region", "")
        severity = finding.get("severity", "unknown").lower()

        if severity in ("severe", "critical"):
            icon = "🔴"
        elif severity == "moderate":
            icon = "🟡"
        elif severity == "mild":
            icon = "🔵"
        elif severity == "normal":
            icon = "🟢"
        else:
            icon = "⚪"

        region_str = f" ({region})" if region else ""
        st.markdown(f"{icon} **{text}**{region_str} — *{severity}*")
