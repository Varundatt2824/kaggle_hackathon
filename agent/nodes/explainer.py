"""
Explainer Node — generates the plain-language summary.

This is where MedSimplify's core value is delivered. The model takes
structured findings and produces a clear, empathetic explanation that
a non-medical person can understand.

Why is this separate from the analyzer?
Separation of concerns: the analyzer's job is accurate extraction,
the explainer's job is clear communication. Different skills,
different prompts, different temperature settings (extraction needs
precision at temp=0, explanation benefits from slight creativity at temp=0.3).
"""

import json

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agent.config import settings
from agent.prompts.templates import EXPLAINER_PROMPT, SAFETY_PREAMBLE
from agent.state import MedSimplifyState


DISCLAIMER = (
    "\n\n---\n"
    "**Disclaimer:** This explanation is for educational purposes only and is "
    "not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult your healthcare provider with questions about your results."
)


def explainer_node(state: MedSimplifyState) -> dict:
    """Generate a plain-language explanation of the findings."""
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,  # Slight creativity for natural, empathetic language
    )

    # Use detected language as default if user hasn't specified a preference
    output_language = state.get("output_language") or state.get("detected_language", "English")

    prompt = EXPLAINER_PROMPT.format(
        safety_preamble=SAFETY_PREAMBLE,
        report_type=state["report_type"],
        findings=json.dumps(state["findings"], indent=2),
        output_language=output_language,
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    # Append the safety disclaimer to every explanation
    explanation = response.content + DISCLAIMER

    return {"explanation": explanation}
