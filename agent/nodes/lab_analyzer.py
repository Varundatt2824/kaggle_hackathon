"""
Lab Analyzer Node — extracts structured findings from laboratory reports.

This node asks Gemma 4 to parse a lab report into structured JSON.
Why JSON output? Structured data lets the explainer node reference
specific values ("Your hemoglobin is 13.5, which is normal") rather
than just paraphrasing the whole report. It also enables the UI to
show color-coded indicators (green for normal, red for abnormal).
"""

import json

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agent.config import settings
from agent.prompts.templates import LAB_ANALYZER_PROMPT, SAFETY_PREAMBLE
from agent.state import MedSimplifyState


def lab_analyzer_node(state: MedSimplifyState) -> dict:
    """Extract structured findings from a lab report."""
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,  # Deterministic for structured extraction
    )

    prompt = LAB_ANALYZER_PROMPT.format(
        safety_preamble=SAFETY_PREAMBLE,
        extracted_text=state["extracted_text"],
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    findings = _parse_findings(response.content)

    return {"findings": findings}


def _parse_findings(raw_response: str) -> list[dict]:
    """Parse the LLM's JSON response into a list of findings.

    Why a separate parsing function?
    LLMs don't always return perfect JSON — they might wrap it in
    markdown code blocks or add commentary. This function handles
    those edge cases gracefully rather than crashing.
    """
    text = raw_response.strip()

    # Strip markdown code block if present (```json ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        text = "\n".join(lines[1:-1])

    try:
        findings = json.loads(text)
        if isinstance(findings, list):
            return findings
        return [findings]
    except json.JSONDecodeError:
        # If JSON parsing fails, return a single finding with the raw text
        # so the pipeline doesn't crash — the explainer can still work with this
        return [{"name": "Unparsed Report", "raw_text": text, "status": "unknown"}]
