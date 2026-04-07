"""
Chat Node — handles follow-up conversation about the report.

This is the "loop" part of our LangGraph. After the initial analysis,
the user can ask questions like "What does high LDL mean for me?"
and the model answers grounded in the actual report findings.

Why scope the chat to the report?
An open-ended medical chatbot is dangerous and a completely different
product. By grounding every answer in the report's actual findings,
we reduce hallucination risk and keep the tool focused on its purpose.
"""

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama

from agent.config import settings
from agent.prompts.templates import CHAT_PROMPT, SAFETY_PREAMBLE
from agent.state import MedSimplifyState


def chat_node(state: MedSimplifyState) -> dict:
    """Respond to a follow-up question about the report.

    The chat_history uses a LangGraph reducer (operator.add), so
    we return only the NEW messages to append, not the full history.
    """
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
    )

    output_language = state.get("output_language") or state.get("detected_language", "English")

    system_prompt = CHAT_PROMPT.format(
        safety_preamble=SAFETY_PREAMBLE,
        report_type=state["report_type"],
        findings=json.dumps(state["findings"], indent=2),
        explanation=state["explanation"],
        output_language=output_language,
    )

    # Build the full message list: system context + conversation history
    messages: list[BaseMessage] = [
        HumanMessage(content=system_prompt),
        *state["chat_history"],
    ]

    response = llm.invoke(messages)

    # Return only the new AI message — the reducer will append it
    return {"chat_history": [AIMessage(content=response.content)]}
