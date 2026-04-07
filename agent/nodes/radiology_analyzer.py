"""
Radiology Analyzer Node — extracts structured findings from radiology reports.

Radiology reports are narrative text (written by radiologists), not tables.
The structure is different from lab reports:
- "Findings" section: detailed observations per body region
- "Impression" section: the radiologist's summary/conclusion

This node extracts both into structured JSON for the explainer.
"""

import json

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agent.config import settings
from agent.prompts.templates import RADIOLOGY_ANALYZER_PROMPT, SAFETY_PREAMBLE
from agent.state import MedSimplifyState


def radiology_analyzer_node(state: MedSimplifyState) -> dict:
    """Extract structured findings from a radiology report."""
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

    prompt = RADIOLOGY_ANALYZER_PROMPT.format(
        safety_preamble=SAFETY_PREAMBLE,
        extracted_text=state["extracted_text"],
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    findings = _parse_findings(response.content)

    return {"findings": findings}


def _parse_findings(raw_response: str) -> list[dict]:
    """Parse radiology findings from LLM response."""
    text = raw_response.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        findings = json.loads(text)
        if isinstance(findings, list):
            return findings
        return [findings]
    except json.JSONDecodeError:
        return [{"finding": text, "region": "unknown", "severity": "unknown"}]
