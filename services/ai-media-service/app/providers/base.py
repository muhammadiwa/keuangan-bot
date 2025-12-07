"""
Base classes for AI provider abstraction.

This module defines the core interfaces and data structures used by all AI providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# Health check constants
HEALTH_CHECK_TIMEOUT = 10.0  # seconds
LATENCY_THRESHOLD_MS = 5000.0  # 5 seconds - latency above this is considered slow


@dataclass
class AIResponse:
    """
    Standardized response structure from AI providers.
    
    Attributes:
        content: The text content returned by the AI model
        model: The model identifier used for the request
        provider: The provider name (e.g., 'ollama', 'openai', 'anthropic')
        usage: Token usage statistics (input_tokens, output_tokens)
        latency_ms: Request latency in milliseconds
        raw_response: The original response from the provider (for debugging)
    """
    content: str
    model: str
    provider: str
    usage: dict[str, int] | None = None
    latency_ms: float = 0.0
    raw_response: dict[str, Any] | None = field(default=None, repr=False)


class AIProvider(ABC):
    """
    Abstract base class for all AI providers.
    
    All provider implementations must inherit from this class and implement
    the required abstract methods to ensure consistent behavior across providers.
    """
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Send a chat completion request to the AI provider.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Optional model override (uses default if not specified)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
            
        Returns:
            AIResponse with the model's response and metadata
            
        Raises:
            ProviderError: If the request fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the AI provider.
        
        Returns:
            Dict with health status information including:
            - status: 'healthy', 'degraded', or 'unhealthy'
            - provider: Provider name
            - latency_ms: Response time in milliseconds
            - latency_warning: True if latency exceeds threshold (optional)
            - error: Error message (if unhealthy)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the provider name identifier.
        
        Returns:
            String identifier for the provider (e.g., 'ollama', 'openai')
        """
        pass


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class AuthenticationError(ProviderError):
    """Raised when authentication fails (invalid API key, expired token)."""
    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, provider: str, retry_after: float | None = None):
        super().__init__(message, provider, status_code=429, retryable=True)
        self.retry_after = retry_after


class QuotaExceededError(ProviderError):
    """Raised when quota/credits are exhausted."""
    pass


def build_health_response(
    status: str,
    provider: str,
    latency_ms: float,
    base_url: str | None = None,
    model: str | None = None,
    error: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """
    Build a standardized health check response with latency threshold evaluation.
    
    Args:
        status: Base status ('healthy' or 'unhealthy')
        provider: Provider name
        latency_ms: Response latency in milliseconds
        base_url: Provider base URL (optional)
        model: Model name (optional)
        error: Error message if unhealthy (optional)
        **extra: Additional fields to include in response
        
    Returns:
        Dict with standardized health response including latency warning if applicable
    """
    response: dict[str, Any] = {
        "status": status,
        "provider": provider,
        "latency_ms": round(latency_ms, 2),
    }
    
    if base_url:
        response["base_url"] = base_url
    
    if model:
        response["model"] = model
    
    # Add latency warning if threshold exceeded
    if status == "healthy" and latency_ms > LATENCY_THRESHOLD_MS:
        response["latency_warning"] = True
        response["latency_threshold_ms"] = LATENCY_THRESHOLD_MS
    
    if error:
        response["error"] = error
    
    # Add any extra fields
    response.update(extra)
    
    return response
