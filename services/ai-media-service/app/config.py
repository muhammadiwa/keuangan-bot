from functools import lru_cache
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Supported AI providers
AIProviderType = Literal[
    "ollama",
    "openai",
    "megallm",
    "groq",
    "together",
    "deepseek",
    "qwen",
    "kimi",
    "moonshot",
    "glm",
    "zhipu",
    "bigmodel",
    "anthropic",
    "gemini",
]

# Supported STT providers
STTProviderType = Literal["whisper", "openai", "groq"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Legacy Ollama settings (kept for backward compatibility)
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:3b-instruct"

    # Primary AI Provider settings
    ai_provider: AIProviderType = "ollama"
    ai_api_key: str | None = None
    ai_base_url: str | None = None
    ai_model: str | None = None

    # Fallback AI Provider settings
    ai_fallback_provider: AIProviderType | None = None
    ai_fallback_api_key: str | None = None
    ai_fallback_base_url: str | None = None
    ai_fallback_model: str | None = None

    # STT Provider settings
    stt_provider: STTProviderType = "whisper"
    stt_api_key: str | None = None
    stt_model: str = "base"

    # Legacy whisper settings (kept for backward compatibility)
    whisper_model: str = "base"

    # OCR settings
    tesseract_lang: str = "eng+ind"

    # Service settings
    port: int = 8500

    # AI Options
    ai_timeout: float = 30.0
    ai_max_retries: int = 2
    ai_debug_logging: bool = False

    # Backend API settings (for cost tracking)
    backend_api_url: str = "http://backend-api:8000"
    ai_cost_tracking_enabled: bool = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    @field_validator("ai_fallback_provider", mode="before")
    @classmethod
    def normalize_fallback_provider(cls, v: str | None) -> str | None:
        """Normalize fallback provider names to lowercase, treating empty strings as None."""
        if v is None or v == "":
            return None
        return v.lower()

    @field_validator("ai_provider", mode="before")
    @classmethod
    def normalize_provider(cls, v: str | None) -> str:
        """Normalize provider names to lowercase, defaulting empty to 'ollama'."""
        if v is None or v == "":
            return "ollama"
        return v.lower()

    @field_validator("stt_provider", mode="before")
    @classmethod
    def normalize_stt_provider(cls, v: str | None) -> str | None:
        """Normalize STT provider names to lowercase."""
        if v is None:
            return "whisper"
        return v.lower()

    def get_effective_ai_model(self) -> str:
        """Get the effective AI model, falling back to ollama_model if not set."""
        if self.ai_model:
            return self.ai_model
        if self.ai_provider == "ollama":
            return self.ollama_model
        return "gpt-3.5-turbo"  # Default for cloud providers

    def get_effective_ai_base_url(self) -> str | None:
        """Get the effective AI base URL, falling back to ollama_url if not set."""
        if self.ai_base_url:
            return self.ai_base_url
        if self.ai_provider == "ollama":
            return self.ollama_url
        return None  # Use provider defaults

    def get_effective_stt_model(self) -> str:
        """Get the effective STT model, falling back to whisper_model if not set."""
        if self.stt_model:
            return self.stt_model
        return self.whisper_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
