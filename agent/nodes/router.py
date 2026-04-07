"""
Router Node — classifies the report as "lab" or "radiology".

Why a router instead of one generic analyzer?
Lab reports and radiology reports have fundamentally different structures:
- Lab reports: structured tables (test name, value, reference range)
- Radiology reports: narrative text (findings, impressions, conclusions)

Separate analyzers can use specialized prompts and extraction logic
for each format, producing much better results than a one-size-fits-all approach.

This router is the "branching point" in our LangGraph — it's what makes
this a graph (with conditional edges) rather than a simple chain.
"""

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agent.config import settings
from agent.prompts.templates import ROUTER_PROMPT
from agent.state import MedSimplifyState


def router_node(state: MedSimplifyState) -> dict:
    """Classify the report type using Gemma 4."""
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,  # Zero temperature = deterministic output for classification
    )

    prompt = ROUTER_PROMPT.format(extracted_text=state["extracted_text"])
    response = llm.invoke([HumanMessage(content=prompt)])

    report_type = response.content.strip().lower()

    # Defensive: if the model returns something unexpected, default to "lab"
    if report_type not in ("lab", "radiology"):
        report_type = "lab"

    return {"report_type": report_type}


def route_by_type(state: MedSimplifyState) -> str:
    """Routing function used by LangGraph's conditional_edge.

    This function doesn't modify state — it just returns a string
    that tells LangGraph which node to go to next. LangGraph uses
    this with add_conditional_edges() to create the branching logic.
    """
    return state["report_type"]
