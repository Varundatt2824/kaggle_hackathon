# MedSimplify Docker Setup
#
# Runs FastAPI backend + Streamlit frontend in one container.
# Ollama must run on the HOST machine (needs GPU access).
# Docker connects to host Ollama via OLLAMA_BASE_URL.
#
# Build:  docker build -t medsimplify .
# Run:    docker run -p 8000:8000 -p 8501:8501 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 medsimplify

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (Docker caches this layer if deps don't change)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

EXPOSE 8000 8501

# Start both services
CMD ["sh", "-c", "uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 & uv run streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0"]
