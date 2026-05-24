import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    GROQ_API_KEY: str
    TAVILY_API_KEY: str
    MAIN_MODEL: str = "openai/gpt-oss-120b"
    FAST_MODEL: str = "openai/gpt-oss-120b"

    # LangSmith tracing
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "task-a-user-modeling"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"


settings = Settings()

# LangChain reads these directly from os.environ, not from our settings object.
# Propagate so tracing activates without any extra setup in each module.
if settings.LANGCHAIN_TRACING_V2.lower() == "true" and settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
