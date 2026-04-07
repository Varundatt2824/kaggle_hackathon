"""
FastAPI application entry point.

Why FastAPI over Flask?
- Async-first (better for LLM workloads with long response times)
- Automatic API docs at /docs (Swagger) and /redoc
- Pydantic integration for request validation
- Modern Python type hints throughout
- Significantly faster than Flask

Run with: uv run uvicorn api.main:app --reload --port 8000
Then visit: http://localhost:8000/docs to see the interactive API docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.report import router as report_router

app = FastAPI(
    title="MedSimplify API",
    description="Medical Report Interpreter — explains lab reports and radiology summaries in plain language",
    version="0.1.0",
)

# CORS middleware: allows the Streamlit frontend (running on a different port)
# to make requests to this API. Without this, the browser blocks cross-origin requests.
# In production, you'd restrict origins to your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(report_router)


@app.get("/health")
async def health_check():
    """Simple health check endpoint.

    Useful for deployment platforms (Modal, HF Spaces) to verify
    the service is running. Also useful during development to
    confirm the API is up before starting the frontend.
    """
    return {"status": "healthy", "service": "medsimplify-api"}
