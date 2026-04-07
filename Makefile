.PHONY: setup api app dev test clean

# First-time setup
setup:
	uv sync
	cp -n .env.example .env || true
	@echo "Setup complete. Make sure Ollama is running with: ollama run gemma4:31b"

# Run FastAPI backend
api:
	uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit frontend
app:
	uv run streamlit run app/main.py --server.port 8501

# Run both backend and frontend (development)
dev:
	@echo "Starting API server and Streamlit app..."
	@make api & make app

# Run tests
test:
	uv run pytest tests/ -v

# Clean up
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
