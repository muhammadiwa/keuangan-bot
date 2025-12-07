"""
AI Provider abstraction layer for multi-provider support.

This module provides a unified interface for interacting with various AI/LLM providers
including Ollama, OpenAI-compatible APIs, Anthropic, Gemini, and GLM.
"""

from .anthropic import AnthropicProvider
from .base import (
    AIProvider,
    AIResponse,
    AuthenticationError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
)
from .gemini import GeminiProvider
from .glm import GLMProvider
from .manager import AIProviderManager
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider

__all__ = [
    "AIProvider",
    "AIProviderManager",
    "AIResponse",
    "AnthropicProvider",
    "AuthenticationError",
    "GeminiProvider",
    "GLMProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
    "QuotaExceededError",
    "RateLimitError",
]
