"""
Modal deployment for MedSimplify API.

What is Modal?
Modal is a serverless GPU platform. You define your app as Python code,
and Modal handles provisioning GPU machines, building Docker images,
scaling up/down, and routing requests. You pay only when code is running.

How this works:
1. We define a Modal Image (like a Dockerfile but in Python)
2. We mount our FastAPI app into it
3. Modal exposes it as a public HTTPS endpoint
4. The Streamlit frontend on HF Spaces calls this endpoint

Deploy with: uv run modal deploy deploy/modal_app.py
Test locally: uv run modal serve deploy/modal_app.py  (hot-reload mode)
"""

import modal

# --- Modal App definition ---
app = modal.App("medsimplify")

# --- Build the container image ---
# This is like a Dockerfile but written in Python.
# Modal caches each layer, so rebuilds are fast.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi",
        "uvicorn",
        "langchain-core",
        "langchain-ollama",
        "langgraph",
        "pydantic",
        "pydantic-settings",
        "pymupdf",
        "python-multipart",
        "langdetect",
        "babel",
        # For running Ollama inside the container
        "ollama",
    )
    # Install Ollama binary
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
    )
    # Copy our application code into the container
    .add_local_dir("agent", remote_path="/app/agent")
    .add_local_dir("api", remote_path="/app/api")
    .add_local_file(".env.example", remote_path="/app/.env")
)


# --- Ollama model setup ---
# This runs ONCE when the container starts (not per-request).
# The model gets cached in the container's filesystem.
def download_model():
    """Download the Gemma 4 model inside the Modal container."""
    import subprocess
    import time

    # Start Ollama server
    process = subprocess.Popen(["ollama", "serve"])
    time.sleep(5)  # Wait for server to start

    # Pull the model
    subprocess.run(["ollama", "pull", "gemma4:e4b"], check=True)

    # Stop the server (it will restart when we need it)
    process.terminate()


# --- FastAPI web endpoint ---
@app.function(
    image=image,
    gpu="T4",  # NVIDIA T4 — cheapest GPU on Modal, sufficient for gemma4:e4b
    timeout=600,
    # keep_warm=1 keeps one container alive to avoid cold starts.
    # Remove this to save credits (but first request will be slow).
    # keep_warm=1,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI app on Modal.

    @modal.asgi_app() tells Modal this function returns an ASGI app
    (FastAPI is ASGI-based). Modal handles routing HTTP requests to it.
    """
    import subprocess
    import sys
    import time

    # Add app directory to Python path
    sys.path.insert(0, "/app")

    # Start Ollama server in the background
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    # Import and return the FastAPI app
    from api.main import app

    return app
