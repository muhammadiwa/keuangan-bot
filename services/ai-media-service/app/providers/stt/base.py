"""
Base classes for STT (Speech-to-Text) provider abstraction.

This module defines the core interfaces and data structures used by all STT providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# Health check constants
HEALTH_CHECK_TIMEOUT = 10.0  # seconds


@dataclass
class STTResponse:
    """
    Standardized response structure from STT providers.
    
    Attributes:
        text: The transcribed text
        language: Detected or specified language code
        provider: The provider name (e.g., 'whisper', 'openai', 'groq')
        model: The model used for transcription
        duration_seconds: Audio duration in seconds (if available)
        latency_ms: Request latency in milliseconds
        confidence: Transcription confidence score (0.0 to 1.0, if available)
        raw_response: The original response from the provider (for debugging)
    """
    text: str
    language: str | None = None
    provider: str = ""
    model: str = ""
    duration_seconds: float | None = None
    latency_ms: float = 0.0
    confidence: float | None = None
    raw_response: dict[str, Any] | None = field(default=None, repr=False)


class STTProvider(ABC):
    """
    Abstract base class for all STT providers.
    
    All provider implementations must inherit from this class and implement
    the required abstract methods to ensure consistent behavior across providers.
    """
    
    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> STTResponse:
        """
        Transcribe audio to text.
        
        Args:
            audio_bytes: Raw audio data (supports mp3, wav, m4a, webm, etc.)
            language: Optional language hint (ISO 639-1 code, e.g., 'id', 'en')
            prompt: Optional prompt to guide transcription
            
        Returns:
            STTResponse with the transcribed text and metadata
            
        Raises:
            STTProviderError: If the transcription fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the STT provider.
        
        Returns:
            Dict with health status information including:
            - status: 'healthy', 'degraded', or 'unhealthy'
            - provider: Provider name
            - model: Model name
            - error: Error message (if unhealthy)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the provider name identifier.
        
        Returns:
            String identifier for the provider (e.g., 'whisper', 'openai', 'groq')
        """
        pass


class STTProviderError(Exception):
    """Base exception for STT provider-related errors."""
    
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


class STTAuthenticationError(STTProviderError):
    """Raised when authentication fails (invalid API key)."""
    pass


class STTRateLimitError(STTProviderError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, provider: str, retry_after: float | None = None):
        super().__init__(message, provider, status_code=429, retryable=True)
        self.retry_after = retry_after


def build_stt_health_response(
    status: str,
    provider: str,
    model: str | None = None,
    error: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """
    Build a standardized health check response for STT providers.
    
    Args:
        status: Base status ('healthy' or 'unhealthy')
        provider: Provider name
        model: Model name (optional)
        error: Error message if unhealthy (optional)
        **extra: Additional fields to include in response
        
    Returns:
        Dict with standardized health response
    """
    response: dict[str, Any] = {
        "status": status,
        "provider": provider,
    }
    
    if model:
        response["model"] = model
    
    if error:
        response["error"] = error
    
    # Add any extra fields
    response.update(extra)
    
    return response
