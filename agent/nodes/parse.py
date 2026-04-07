"""
Parse Node — extracts text from PDF, image, or raw text input.

Why is this the first node?
All downstream nodes work with text. This node normalizes any input
format into plain text. For images, we use Gemma 4's multimodal
vision — the model "reads" the image directly, which is more reliable
than traditional OCR for messy medical report photos.
"""

import base64

import fitz  # PyMuPDF — imported as 'fitz' for historical reasons (named after the Fitz graphics library)
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langdetect import detect, LangDetectException

from agent.config import settings
from agent.prompts.templates import PARSE_IMAGE_PROMPT
from agent.state import MedSimplifyState


def parse_node(state: MedSimplifyState) -> dict:
    """Extract text from the input document and detect its language.

    Returns a dict (not full state) — LangGraph merges this into the state.
    This is a key LangGraph pattern: nodes return ONLY the fields they update.
    """
    input_type = state["input_type"]
    raw_input = state["raw_input"]

    if input_type == "text":
        extracted = raw_input

    elif input_type == "pdf":
        extracted = _extract_from_pdf(raw_input)

    elif input_type == "image":
        extracted = _extract_from_image(raw_input)

    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    detected_language = _detect_language(extracted)

    return {"extracted_text": extracted, "detected_language": detected_language}


def _extract_from_pdf(raw_input: str) -> str:
    """Extract text from a base64-encoded PDF using PyMuPDF.

    Why PyMuPDF over pdfplumber or PyPDF2?
    - Fastest pure-Python PDF library
    - No system dependencies (unlike poppler)
    - Handles most lab report PDFs well (they're usually text-based, not scanned)
    """
    pdf_bytes = base64.b64decode(raw_input)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    return "\n".join(text_parts)


def _extract_from_image(raw_input: str) -> str:
    """Use Gemma 4's multimodal vision to read text from an image.

    Why Gemma 4 vision instead of Tesseract OCR?
    - Gemma 4 understands context, not just characters — it knows what a
      "reference range" looks like even if the image is blurry
    - No need for tesseract system dependency
    - Aligns with the hackathon goal of showcasing Gemma 4's capabilities
    """
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": PARSE_IMAGE_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{raw_input}"},
            },
        ]
    )

    response = llm.invoke([message])
    return response.content


def _detect_language(text: str) -> str:
    """Detect the language of the extracted text.

    Why langdetect instead of an LLM call?
    It's instant (no model inference), works offline, and supports
    55 languages out of the box. No point burning GPU cycles for this.

    We use the `pycountry` approach — the ISO 639-1 code from langdetect
    is converted to the full language name dynamically, so every language
    in the world is covered without maintaining a manual mapping.
    """
    try:
        code = detect(text)
    except LangDetectException:
        return "English"

    # Convert ISO 639 code to full language name using babel,
    # which covers all ISO 639 languages (~7000+).
    try:
        from babel import Locale
        # langdetect can return codes like "zh-cn" — normalize to babel format
        locale_obj = Locale.parse(code.replace("-", "_"))
        return locale_obj.get_language_name("en")
    except Exception:
        # Fallback: return the raw code capitalized
        return code.capitalize()
