"""
Central configuration for MedSimplify.

Why a config module? Instead of scattering os.getenv() calls everywhere,
we use Pydantic's BaseSettings to load env vars once, validate them,
and provide typed access. If a required setting is missing, you get a
clear error at startup — not a mysterious crash later.
"""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:31b"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
