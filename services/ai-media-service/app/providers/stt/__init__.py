"""
Speech-to-Text (STT) Provider implementations.

This module provides a unified interface for multiple STT providers:
- Local Whisper (faster-whisper)
- OpenAI Whisper API
- Groq Whisper API

Usage:
    from app.providers.stt import STTProviderManager
    
    manager = STTProviderManager(settings)
    result = await manager.transcribe(audio_bytes)
"""

from .base import STTProvider, STTResponse, STTProviderError
from .manager import STTProviderManager
from .whisper_local import WhisperLocalProvider
from .openai_whisper import OpenAIWhisperProvider
from .groq_whisper import GroqWhisperProvider

__all__ = [
    "STTProvider",
    "STTResponse",
    "STTProviderError",
    "STTProviderManager",
    "WhisperLocalProvider",
    "OpenAIWhisperProvider",
    "GroqWhisperProvider",
]
