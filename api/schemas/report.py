"""
API request/response schemas.

Why Pydantic models for API schemas?
FastAPI uses Pydantic under the hood for:
1. Automatic request validation (wrong type? missing field? → 422 error)
2. Auto-generated API docs (Swagger UI at /docs)
3. Serialization — converts Python objects to JSON automatically

These schemas are the "contract" between frontend and backend.
The frontend knows exactly what to send and what to expect back.
"""

from pydantic import BaseModel, Field


# --- Request schemas ---

class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint.

    Why base64 for files instead of multipart form upload?
    Streamlit's file_uploader gives us bytes, and sending JSON with
    base64-encoded content is simpler than multipart. For a production
    app you'd use multipart uploads, but for this project JSON is cleaner.
    """
    raw_input: str = Field(description="Report text, or base64-encoded PDF/image")
    input_type: str = Field(description="'text', 'pdf', or 'image'")
    file_name: str = Field(default="", description="Original file name")
    output_language: str = Field(default="", description="Preferred output language (empty = auto-detect)")


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    message: str = Field(description="User's follow-up question")
    # We need the full analysis context to ground the chat response
    report_type: str
    findings: list[dict]
    explanation: str
    output_language: str = Field(default="English")
    chat_history: list[dict] = Field(default_factory=list, description="Previous chat messages as {role, content} dicts")


# --- Response schemas ---

class AnalyzeResponse(BaseModel):
    """Response from the /analyze endpoint."""
    report_type: str
    findings: list[dict]
    explanation: str
    followup_questions: list[str]
    detected_language: str


class ChatResponse(BaseModel):
    """Response from the /chat endpoint."""
    response: str
    chat_history: list[dict]


class TranslateRequest(BaseModel):
    """Request body for the /translate endpoint."""
    text: str = Field(description="The explanation text to translate")
    followup_questions: list[str] = Field(default_factory=list)
    target_language: str = Field(description="Target language name, e.g., 'Hindi', 'Spanish'")


class TranslateResponse(BaseModel):
    """Response from the /translate endpoint."""
    translated_text: str
    translated_questions: list[str]
