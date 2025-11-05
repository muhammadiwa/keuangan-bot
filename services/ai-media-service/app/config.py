from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ollama_url: str
    ollama_model: str
    whisper_model: str
    tesseract_lang: str
    port: int

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


@lru_cache
def get_settings() -> Settings:
    return Settings()
