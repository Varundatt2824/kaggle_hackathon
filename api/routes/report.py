"""
Report analysis API routes.

Why a separate router file instead of putting everything in main.py?
As your API grows, a single file becomes unmanageable. FastAPI's
APIRouter lets you group related endpoints and include them in the
main app. This is the same pattern as Flask Blueprints.
"""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from langchain_ollama import ChatOllama

from agent.config import settings
from agent.graph import build_analysis_graph, build_chat_graph
from api.schemas.report import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    TranslateRequest,
    TranslateResponse,
)

router = APIRouter(prefix="/api/v1", tags=["report"])

# Build graphs once at module level — they're reusable and stateless.
# The state is passed in per-request, not stored in the graph.
analysis_graph = build_analysis_graph()
chat_graph = build_chat_graph()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_report(request: AnalyzeRequest):
    """Analyze a medical report and return a plain-language explanation.

    Why async? FastAPI supports async endpoints, which means the server
    can handle other requests while waiting for the LLM to respond.
    However, since our LLM calls are synchronous (Ollama), FastAPI
    automatically runs this in a thread pool. We mark it async for
    consistency and future-proofing (Modal deployment will be truly async).
    """
    try:
        initial_state = {
            "raw_input": request.raw_input,
            "input_type": request.input_type,
            "file_name": request.file_name,
            "extracted_text": "",
            "report_type": "",
            "detected_language": "",
            "output_language": request.output_language,
            "findings": [],
            "explanation": "",
            "followup_questions": [],
            "chat_history": [],
        }

        result = analysis_graph.invoke(initial_state)

        return AnalyzeResponse(
            report_type=result["report_type"],
            findings=result["findings"],
            explanation=result["explanation"],
            followup_questions=result["followup_questions"],
            detected_language=result.get("detected_language", "English"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat_followup(request: ChatRequest):
    """Handle a follow-up question about an analyzed report.

    The frontend sends the full analysis context (findings, explanation)
    along with the question. This is stateless on the server — we don't
    store sessions. The frontend holds all state. This makes the API
    simpler and easier to scale.
    """
    try:
        # Convert chat history dicts back to LangChain message objects
        chat_history = []
        for msg in request.chat_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        # Add the new user message
        chat_history.append(HumanMessage(content=request.message))

        chat_state = {
            "raw_input": "",
            "input_type": "",
            "file_name": "",
            "extracted_text": "",
            "report_type": request.report_type,
            "detected_language": "",
            "output_language": request.output_language,
            "findings": request.findings,
            "explanation": request.explanation,
            "followup_questions": [],
            "chat_history": chat_history,
        }

        result = chat_graph.invoke(chat_state)

        # Convert messages back to dicts for JSON serialization
        history_dicts = []
        for msg in result["chat_history"]:
            if isinstance(msg, HumanMessage):
                history_dicts.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_dicts.append({"role": "assistant", "content": msg.content})

        return ChatResponse(
            response=result["chat_history"][-1].content,
            chat_history=history_dicts,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/translate", response_model=TranslateResponse)
async def translate_explanation(request: TranslateRequest):
    """Translate the explanation and follow-up questions into another language.

    Why a separate endpoint instead of re-running the analysis?
    The analysis (parsing, routing, extraction) is expensive and language-independent.
    Translation is just one LLM call on already-processed text — much faster.
    """
    try:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )

        prompt = (
            f"Translate the following medical report explanation into {request.target_language}. "
            f"Keep the same structure, formatting (markdown), and tone. "
            f"Do NOT add or remove any medical information. "
            f"Translate accurately — do not interpret or change meaning.\n\n"
            f"{request.text}"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        translated_text = response.content

        # Translate follow-up questions
        translated_questions = []
        if request.followup_questions:
            questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(request.followup_questions))
            q_prompt = (
                f"Translate these questions into {request.target_language}. "
                f"Return only the translated numbered list.\n\n{questions_text}"
            )
            q_response = llm.invoke([HumanMessage(content=q_prompt)])
            # Parse numbered list back
            for line in q_response.content.strip().split("\n"):
                line = line.strip()
                if line:
                    # Remove numbering prefix
                    for sep in [". ", ") ", ": "]:
                        for n in range(1, 10):
                            prefix = f"{n}{sep}"
                            if line.startswith(prefix):
                                line = line[len(prefix):]
                                break
                    if line:
                        translated_questions.append(line)

        return TranslateResponse(
            translated_text=translated_text,
            translated_questions=translated_questions,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
