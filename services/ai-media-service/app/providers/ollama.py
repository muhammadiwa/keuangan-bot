"""
Ollama AI Provider implementation.

This module provides integration with Ollama for self-hosted AI inference.
Ollama uses a native API format with the /api/generate endpoint.
"""

import time
from typing import Any

import httpx
from loguru import logger

from .base import AIProvider, AIResponse, ProviderError


class OllamaProvider(AIProvider):
    """
    AI Provider implementation for Ollama.
    
    Ollama is a self-hosted LLM runtime that supports various open-source models.
    It uses a native API format different from OpenAI-compatible APIs.
    
    Attributes:
        base_url: The Ollama server URL (default: http://ollama:11434)
        model: The model to use (e.g., qwen2.5:3b-instruct, llama3, phi3)
        timeout: Request timeout in seconds
    """
    
    DEFAULT_BASE_URL = "http://ollama:11434"
    DEFAULT_MODEL = "qwen2.5:3b-instruct"
    
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the Ollama provider.
        
        Args:
            base_url: Ollama server URL (defaults to http://ollama:11434)
            model: Model name to use (defaults to qwen2.5:3b-instruct)
            timeout: Request timeout in seconds (defaults to 120.0)
        """
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        """Return the provider name identifier."""
        return "ollama"
    
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Send a chat completion request to Ollama.
        
        Ollama uses the /api/generate endpoint with a prompt format.
        Messages are converted to a single prompt string.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Optional model override
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response (maps to num_predict)
            
        Returns:
            AIResponse with the model's response
            
        Raises:
            ProviderError: If the request fails
        """
        effective_model = model or self.model
        prompt = self._messages_to_prompt(messages)
        
        payload: dict[str, Any] = {
            "model": effective_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        
        logger.debug(
            "Sending request to Ollama",
            model=effective_model,
            base_url=self.base_url,
            prompt_length=len(prompt),
        )
        
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as e:
            raise ProviderError(
                message=f"Ollama request timed out after {self.timeout}s",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.ConnectError as e:
            raise ProviderError(
                message=f"Failed to connect to Ollama at {self.base_url}",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"Ollama returned error: {e.response.status_code}",
                provider=self.name,
                status_code=e.response.status_code,
                retryable=e.response.status_code >= 500,
            ) from e
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        content = data.get("response", "")
        
        # Extract usage info if available
        usage = None
        if "prompt_eval_count" in data or "eval_count" in data:
            usage = {
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            }
        
        logger.debug(
            "Ollama response received",
            model=effective_model,
            latency_ms=round(latency_ms, 2),
            response_length=len(content),
            usage=usage,
        )
        
        return AIResponse(
            content=content,
            model=data.get("model", effective_model),
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the Ollama server.
        
        Calls the /api/tags endpoint to verify connectivity and list models.
        
        Returns:
            Dict with health status information including latency metrics
        """
        from .base import HEALTH_CHECK_TIMEOUT, build_health_response
        
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            models = [m.get("name") for m in data.get("models", [])]
            model_available = self.model in models or any(
                self.model.split(":")[0] in m for m in models
            )
            
            return build_health_response(
                status="healthy",
                provider=self.name,
                latency_ms=latency_ms,
                base_url=self.base_url,
                model=self.model,
                model_available=model_available,
                available_models=models,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Ollama health check failed",
                error=str(e),
                base_url=self.base_url,
            )
            return build_health_response(
                status="unhealthy",
                provider=self.name,
                latency_ms=latency_ms,
                base_url=self.base_url,
                model=self.model,
                error=str(e),
            )
    
    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """
        Convert chat messages to a single prompt string for Ollama.
        
        Ollama's /api/generate endpoint expects a single prompt string,
        so we need to format the messages appropriately.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
        if not messages:
            return ""
        
        # If there's only one message, return its content directly
        if len(messages) == 1:
            return messages[0].get("content", "")
        
        # Format multiple messages with role prefixes
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:  # user or other
                parts.append(f"User: {content}")
        
        return "\n\n".join(parts)
