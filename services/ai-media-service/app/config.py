from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "phi3"
    whisper_model: str = "base"
    tesseract_lang: str = "eng"
    port: int = 8500


@lru_cache
def get_settings() -> Settings:
    return Settings()
