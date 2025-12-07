"""
STT Provider Manager for managing multiple STT providers.

This module provides a unified interface for selecting and using
different STT providers based on configuration.
"""

from typing import Any, TYPE_CHECKING

from loguru import logger

from .base import (
    STTProvider,
    STTResponse,
    STTProviderError,
)
from .whisper_local import WhisperLocalProvider
from .openai_whisper import OpenAIWhisperProvider
from .groq_whisper import GroqWhisperProvider

if TYPE_CHECKING:
    from app.config import Settings


class STTProviderManager:
    """
    Manager for STT provider selection and routing.
    
    This class handles:
    - Provider instantiation based on configuration
    - Request routing to the appropriate provider
    - Health status aggregation
    
    Attributes:
        settings: Application settings
        provider: The active STT provider instance
    """
    
    # Mapping of provider names to their classes
    PROVIDER_CLASSES: dict[str, type[STTProvider]] = {
        "whisper": WhisperLocalProvider,
        "openai": OpenAIWhisperProvider,
        "groq": GroqWhisperProvider,
    }
    
    def __init__(self, settings: "Settings"):
        """
        Initialize the STT Provider Manager.
        
        Args:
            settings: Application settings containing STT configuration
        """
        self.settings = settings
        self._provider: STTProvider | None = None
        self._initialize_provider()
    
    def _initialize_provider(self) -> None:
        """
        Initialize the STT provider based on configuration.
        
        Creates the appropriate provider instance based on STT_PROVIDER setting.
        Falls back to local Whisper if configuration is invalid.
        """
        provider_name = self.settings.stt_provider.lower()
        
        logger.info(
            "Initializing STT provider",
            provider=provider_name,
            model=self.settings.get_effective_stt_model(),
        )
        
        try:
            if provider_name == "whisper":
                self._provider = self._create_whisper_local()
            elif provider_name == "openai":
                self._provider = self._create_openai_whisper()
            elif provider_name == "groq":
                self._provider = self._create_groq_whisper()
            else:
                logger.warning(
                    "Unknown STT provider, falling back to local Whisper",
                    provider=provider_name,
                )
                self._provider = self._create_whisper_local()
                
        except Exception as e:
            logger.error(
                "Failed to initialize STT provider, falling back to local Whisper",
                provider=provider_name,
                error=str(e),
            )
            self._provider = self._create_whisper_local()
    
    def _create_whisper_local(self) -> WhisperLocalProvider:
        """Create a local Whisper provider instance."""
        return WhisperLocalProvider(
            model_name=self.settings.get_effective_stt_model(),
            device="cpu",
            compute_type="int8",
        )
    
    def _create_openai_whisper(self) -> OpenAIWhisperProvider:
        """
        Create an OpenAI Whisper provider instance.
        
        Uses STT_API_KEY for authentication, falls back to AI_API_KEY
        if STT_API_KEY is not set and AI_PROVIDER is 'openai'.
        """
        api_key = self.settings.stt_api_key
        
        # Fall back to AI_API_KEY if STT_API_KEY not set and using OpenAI for AI
        if not api_key and self.settings.ai_provider == "openai":
            api_key = self.settings.ai_api_key
        
        if not api_key:
            raise STTProviderError(
                message="STT_API_KEY is required for OpenAI Whisper provider",
                provider="openai",
            )
        
        return OpenAIWhisperProvider(
            api_key=api_key,
            model="whisper-1",
            timeout=self.settings.ai_timeout,
        )
    
    def _create_groq_whisper(self) -> GroqWhisperProvider:
        """
        Create a Groq Whisper provider instance.
        
        Uses STT_API_KEY for authentication, falls back to AI_API_KEY
        if STT_API_KEY is not set and AI_PROVIDER is 'groq'.
        """
        api_key = self.settings.stt_api_key
        
        # Fall back to AI_API_KEY if STT_API_KEY not set and using Groq for AI
        if not api_key and self.settings.ai_provider == "groq":
            api_key = self.settings.ai_api_key
        
        if not api_key:
            raise STTProviderError(
                message="STT_API_KEY is required for Groq Whisper provider",
                provider="groq",
            )
        
        # Use the configured STT model or default to whisper-large-v3
        model = self.settings.stt_model
        if model in ("base", "small", "medium", "large"):
            # Map local whisper model names to Groq model names
            model = "whisper-large-v3"
        
        return GroqWhisperProvider(
            api_key=api_key,
            model=model,
            timeout=self.settings.ai_timeout,
        )
    
    @property
    def provider(self) -> STTProvider:
        """Get the active STT provider."""
        if self._provider is None:
            raise STTProviderError(
                message="STT provider not initialized",
                provider="unknown",
            )
        return self._provider
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> STTResponse:
        """
        Transcribe audio using the configured STT provider.
        
        Args:
            audio_bytes: Raw audio data
            language: Optional language hint (ISO 639-1 code)
            prompt: Optional prompt to guide transcription
            
        Returns:
            STTResponse with transcribed text
            
        Raises:
            STTProviderError: If transcription fails
        """
        logger.debug(
            "Transcribing audio",
            provider=self.provider.name,
            audio_size=len(audio_bytes),
            language=language,
        )
        
        try:
            result = await self.provider.transcribe(
                audio_bytes=audio_bytes,
                language=language,
                prompt=prompt,
            )
            
            logger.info(
                "Transcription completed",
                provider=result.provider,
                model=result.model,
                latency_ms=round(result.latency_ms, 2),
                text_length=len(result.text),
            )
            
            return result
            
        except STTProviderError:
            raise
        except Exception as e:
            logger.exception("Unexpected error during transcription", error=str(e))
            raise STTProviderError(
                message=f"Transcription failed: {e}",
                provider=self.provider.name,
            ) from e
    
    async def get_health_status(self) -> dict[str, Any]:
        """
        Get the health status of the STT provider.
        
        Returns:
            Dict with health status information
        """
        try:
            provider_health = await self.provider.health_check()
            
            return {
                "status": provider_health.get("status", "unknown"),
                "provider": self.provider.name,
                "details": provider_health,
            }
            
        except Exception as e:
            logger.warning("Failed to get STT health status", error=str(e))
            return {
                "status": "unknown",
                "provider": self.provider.name if self._provider else "unknown",
                "error": str(e),
            }
    
    def get_provider_info(self) -> dict[str, Any]:
        """
        Get information about the configured STT provider.
        
        Returns:
            Dict with provider configuration details
        """
        return {
            "provider": self.provider.name,
            "model": self.settings.get_effective_stt_model(),
            "configured_provider": self.settings.stt_provider,
        }
