"""
Local Whisper STT Provider implementation using faster-whisper.

This provider runs Whisper locally on the CPU for offline transcription.
"""

import asyncio
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TYPE_CHECKING

from loguru import logger

from .base import (
    STTProvider,
    STTResponse,
    STTProviderError,
    build_stt_health_response,
)

if TYPE_CHECKING:  # pragma: no cover
    from faster_whisper import WhisperModel as WhisperModelType
else:
    WhisperModelType = None

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None  # type: ignore


class WhisperLocalProvider(STTProvider):
    """
    STT Provider implementation using local faster-whisper.
    
    This provider runs Whisper models locally on CPU for offline transcription.
    Supports various model sizes: tiny, base, small, medium, large.
    
    Attributes:
        model_name: Whisper model size (tiny, base, small, medium, large)
        device: Compute device (cpu, cuda)
        compute_type: Compute precision (int8, float16, float32)
    """
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Initialize the local Whisper provider.
        
        Args:
            model_name: Whisper model size (default: 'base')
            device: Compute device (default: 'cpu')
            compute_type: Compute precision (default: 'int8')
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Any | None = None
        self._model_loaded = False
    
    @property
    def name(self) -> str:
        """Return the provider name identifier."""
        return "whisper"
    
    def _ensure_model_loaded(self) -> Any:
        """
        Ensure the Whisper model is loaded (lazy loading).
        
        Returns:
            The loaded WhisperModel instance
            
        Raises:
            STTProviderError: If faster-whisper is not installed
        """
        if self._model is not None:
            return self._model
        
        if WhisperModel is None:
            raise STTProviderError(
                message="faster-whisper is not installed. Install with: pip install faster-whisper",
                provider=self.name,
            )
        
        logger.info(
            "Loading Whisper model",
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._model_loaded = True
        
        logger.info("Whisper model loaded successfully", model=self.model_name)
        return self._model
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> STTResponse:
        """
        Transcribe audio using local Whisper model.
        
        Args:
            audio_bytes: Raw audio data
            language: Optional language hint (ISO 639-1 code)
            prompt: Optional prompt to guide transcription
            
        Returns:
            STTResponse with transcribed text
            
        Raises:
            STTProviderError: If transcription fails
        """
        start_time = time.perf_counter()
        
        # Write audio to temp file
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)
        
        loop = asyncio.get_running_loop()
        
        def _run_transcribe() -> tuple[str, str | None, float | None]:
            model = self._ensure_model_loaded()
            
            transcribe_kwargs: dict[str, Any] = {}
            if language:
                transcribe_kwargs["language"] = language
            if prompt:
                transcribe_kwargs["initial_prompt"] = prompt
            
            segments, info = model.transcribe(str(tmp_path), **transcribe_kwargs)
            
            # Collect all segments
            text_parts = [segment.text.strip() for segment in segments]
            full_text = " ".join(part for part in text_parts if part)
            
            detected_language = getattr(info, "language", None)
            duration = getattr(info, "duration", None)
            
            return full_text, detected_language, duration
        
        try:
            text, detected_lang, duration = await loop.run_in_executor(
                None, _run_transcribe
            )
        except Exception as e:
            logger.exception("Whisper transcription failed", error=str(e))
            raise STTProviderError(
                message=f"Whisper transcription failed: {e}",
                provider=self.name,
            ) from e
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover
                logger.warning("Failed to remove temp audio file", path=str(tmp_path))
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        logger.debug(
            "Whisper transcription completed",
            model=self.model_name,
            latency_ms=round(latency_ms, 2),
            text_length=len(text),
            language=detected_lang,
        )
        
        return STTResponse(
            text=text or "(tidak ada transkrip)",
            language=detected_lang,
            provider=self.name,
            model=self.model_name,
            duration_seconds=duration,
            latency_ms=latency_ms,
        )
    
    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the local Whisper provider.
        
        Returns:
            Dict with health status information
        """
        if WhisperModel is None:
            return build_stt_health_response(
                status="unhealthy",
                provider=self.name,
                model=self.model_name,
                error="faster-whisper is not installed",
            )
        
        return build_stt_health_response(
            status="healthy",
            provider=self.name,
            model=self.model_name,
            model_loaded=self._model_loaded,
            device=self.device,
            compute_type=self.compute_type,
        )
