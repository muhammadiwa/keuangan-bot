"""
OpenAI Whisper API STT Provider implementation.

This provider uses the OpenAI Whisper API for cloud-based transcription.
"""

import time
from typing import Any

import httpx
from loguru import logger

from .base import (
    STTProvider,
    STTResponse,
    STTProviderError,
    STTAuthenticationError,
    STTRateLimitError,
    build_stt_health_response,
    HEALTH_CHECK_TIMEOUT,
)


class OpenAIWhisperProvider(STTProvider):
    """
    STT Provider implementation using OpenAI Whisper API.
    
    This provider uses the OpenAI /v1/audio/transcriptions endpoint
    for cloud-based speech-to-text transcription.
    
    Attributes:
        api_key: OpenAI API key
        base_url: API base URL (default: https://api.openai.com)
        model: Whisper model to use (default: whisper-1)
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "whisper-1",
        timeout: float = 60.0,
    ):
        """
        Initialize the OpenAI Whisper provider.
        
        Args:
            api_key: OpenAI API key for authentication
            base_url: Optional custom base URL
            model: Whisper model to use (default: 'whisper-1')
            timeout: Request timeout in seconds (default: 60.0)
            
        Raises:
            STTAuthenticationError: If api_key is missing or empty
        """
        if not api_key or not api_key.strip():
            raise STTAuthenticationError(
                message="API key is required for OpenAI Whisper provider",
                provider="openai",
            )
        
        self.api_key = api_key.strip()
        self.base_url = (base_url or "https://api.openai.com").rstrip("/")
        self.model = model
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        """Return the provider name identifier."""
        return "openai"
    
    def _get_headers(self) -> dict[str, str]:
        """Get the HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
        }
    
    def _get_endpoint(self) -> str:
        """Get the transcription endpoint URL."""
        return f"{self.base_url}/v1/audio/transcriptions"
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> STTResponse:
        """
        Transcribe audio using OpenAI Whisper API.
        
        Args:
            audio_bytes: Raw audio data
            language: Optional language hint (ISO 639-1 code)
            prompt: Optional prompt to guide transcription
            
        Returns:
            STTResponse with transcribed text
            
        Raises:
            STTAuthenticationError: If authentication fails
            STTRateLimitError: If rate limit is exceeded
            STTProviderError: For other API errors
        """
        start_time = time.perf_counter()
        
        # Build multipart form data
        files = {
            "file": ("audio.mp3", audio_bytes, "audio/mpeg"),
        }
        data: dict[str, str] = {
            "model": self.model,
            "response_format": "verbose_json",
        }
        
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        
        logger.debug(
            "Sending request to OpenAI Whisper API",
            model=self.model,
            audio_size=len(audio_bytes),
            language=language,
        )
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self._get_endpoint(),
                    headers=self._get_headers(),
                    files=files,
                    data=data,
                )
                
                if response.status_code == 401:
                    raise STTAuthenticationError(
                        message="Authentication failed: Invalid API key",
                        provider=self.name,
                        status_code=401,
                    )
                elif response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    raise STTRateLimitError(
                        message="Rate limit exceeded for OpenAI Whisper",
                        provider=self.name,
                        retry_after=float(retry_after) if retry_after else None,
                    )
                
                response.raise_for_status()
                result = response.json()
                
        except httpx.TimeoutException as e:
            raise STTProviderError(
                message=f"OpenAI Whisper request timed out after {self.timeout}s",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.ConnectError as e:
            raise STTProviderError(
                message=f"Failed to connect to OpenAI at {self.base_url}",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise STTProviderError(
                message=f"OpenAI Whisper returned error: {e.response.status_code}",
                provider=self.name,
                status_code=e.response.status_code,
                retryable=e.response.status_code >= 500,
            ) from e
        except (STTAuthenticationError, STTRateLimitError):
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse response
        text = result.get("text", "")
        detected_language = result.get("language")
        duration = result.get("duration")
        
        logger.debug(
            "OpenAI Whisper transcription completed",
            model=self.model,
            latency_ms=round(latency_ms, 2),
            text_length=len(text),
            language=detected_language,
        )
        
        return STTResponse(
            text=text or "(tidak ada transkrip)",
            language=detected_language,
            provider=self.name,
            model=self.model,
            duration_seconds=duration,
            latency_ms=latency_ms,
            raw_response=result,
        )
    
    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the OpenAI Whisper provider.
        
        Verifies API key validity by checking the models endpoint.
        
        Returns:
            Dict with health status information
        """
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                # Check models endpoint to verify API key
                response = await client.get(
                    f"{self.base_url}/v1/models",
                    headers=self._get_headers(),
                )
                
                if response.status_code == 401:
                    return build_stt_health_response(
                        status="unhealthy",
                        provider=self.name,
                        model=self.model,
                        error="Authentication failed: Invalid API key",
                    )
                
                response.raise_for_status()
                
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return build_stt_health_response(
                status="healthy",
                provider=self.name,
                model=self.model,
                base_url=self.base_url,
                latency_ms=round(latency_ms, 2),
            )
            
        except Exception as e:
            logger.warning(
                "OpenAI Whisper health check failed",
                error=str(e),
            )
            return build_stt_health_response(
                status="unhealthy",
                provider=self.name,
                model=self.model,
                error=str(e),
            )
