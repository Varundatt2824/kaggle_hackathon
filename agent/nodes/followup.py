"""
Follow-up Node — generates suggested questions for the doctor.

Why suggest questions?
Most patients leave the doctor's office thinking "I should have asked..."
By providing specific, findings-based questions, MedSimplify empowers
patients to have more productive conversations with their healthcare
provider. This directly supports the "Health & Sciences" impact track.
"""

import json

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agent.config import settings
from agent.prompts.templates import FOLLOWUP_PROMPT, SAFETY_PREAMBLE
from agent.state import MedSimplifyState


def followup_node(state: MedSimplifyState) -> dict:
    """Generate suggested follow-up questions for the doctor."""
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
    )

    output_language = state.get("output_language") or state.get("detected_language", "English")

    prompt = FOLLOWUP_PROMPT.format(
        safety_preamble=SAFETY_PREAMBLE,
        report_type=state["report_type"],
        findings=json.dumps(state["findings"], indent=2),
        output_language=output_language,
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse numbered list into a Python list
    questions = _parse_questions(response.content)

    return {"followup_questions": questions}


def _parse_questions(raw_response: str) -> list[str]:
    """Parse a numbered list response into a list of strings."""
    lines = raw_response.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove numbering (1. 2. 3. or 1) 2) 3))
        for prefix_len in range(1, 4):
            for separator in [". ", ") ", ": "]:
                prefix = f"{prefix_len}{separator}"
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
        if line:
            questions.append(line)
    return questions
