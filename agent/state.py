"""
MedSimplify Agent State.

This is the "contract" between all nodes in the LangGraph agent.
Every node receives this state, reads what it needs, and writes its
outputs back. LangGraph handles merging the updates automatically.

Why TypedDict over a dataclass or Pydantic model?
LangGraph requires TypedDict for state — it uses the type annotations
to know which fields each node can update. It also supports "reducers"
(like operator.add for lists) to handle how updates are merged.
"""

import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class MedSimplifyState(TypedDict):
    """State that flows through the entire agent graph.

    Fields are grouped by pipeline stage for clarity.
    """

    # --- Input stage ---
    raw_input: str  # Original text content or base64-encoded file
    input_type: str  # "pdf", "image", or "text"
    file_name: str  # Original file name (for display)

    # --- Parse stage ---
    extracted_text: str  # Text extracted from the document

    # --- Router stage ---
    report_type: str  # "lab" or "radiology"

    # --- Language ---
    detected_language: str  # Auto-detected language of the document
    output_language: str  # User's preferred output language (default: same as detected)

    # --- Analysis stage ---
    findings: list[dict]  # Structured findings from the analyzer
    # Lab example:  {"name": "Hemoglobin", "value": "13.5", "unit": "g/dL",
    #                "ref_range": "12.0-17.5", "status": "normal"}
    # Radiology example: {"finding": "No acute abnormality", "region": "chest",
    #                     "severity": "normal"}

    # --- Explanation stage ---
    explanation: str  # Plain-language summary of findings
    followup_questions: list[str]  # Suggested questions for the doctor

    # --- Chat stage ---
    # Annotated with operator.add so new messages are APPENDED, not replaced.
    # This is a LangGraph "reducer" — without it, each node would overwrite
    # the entire chat history. With it, nodes just add new messages.
    chat_history: Annotated[list[BaseMessage], operator.add]
