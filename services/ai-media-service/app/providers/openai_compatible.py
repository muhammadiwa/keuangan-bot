"""
OpenAI-Compatible AI Provider implementation.

This module provides integration with OpenAI-compatible APIs including:
- OpenAI
- MegaLLM
- Groq
- Together AI
- Deepseek
- Qwen (Alibaba DashScope)
- Kimi/Moonshot

All these providers use the standard OpenAI API format with POST /v1/chat/completions.
"""

import time
from typing import Any

import httpx
from loguru import logger

from .base import (
    AIProvider,
    AIResponse,
    AuthenticationError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
)


# Default base URLs for each provider
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com",
    "megallm": "https://api.megallm.id/v1",
    "groq": "https://api.groq.com/openai",
    "together": "https://api.together.xyz",
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode",
    "kimi": "https://api.moonshot.cn",
    "moonshot": "https://api.moonshot.cn",
}

# Default models for each provider
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-3.5-turbo",
    "megallm": "gpt-3.5-turbo",
    "groq": "llama-3.1-8b-instant",
    "together": "meta-llama/Llama-3-8b-chat-hf",
    "deepseek": "deepseek-chat",
    "qwen": "qwen-turbo",
    "kimi": "moonshot-v1-8k",
    "moonshot": "moonshot-v1-8k",
}


class OpenAICompatibleProvider(AIProvider):
    """
    AI Provider implementation for OpenAI-compatible APIs.
    
    This provider supports multiple services that implement the OpenAI API format:
    - OpenAI (api.openai.com)
    - MegaLLM (api.megallm.id)
    - Groq (api.groq.com)
    - Together AI (api.together.xyz)
    - Deepseek (api.deepseek.com)
    - Qwen/DashScope (dashscope.aliyuncs.com)
    - Kimi/Moonshot (api.moonshot.cn)
    
    All use Bearer token authentication and POST /v1/chat/completions endpoint.
    
    Attributes:
        provider_name: The specific provider name (e.g., 'openai', 'groq')
        base_url: The API base URL
        api_key: The API key for authentication
        model: The model to use
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        provider_name: str,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the OpenAI-compatible provider.
        
        Args:
            provider_name: Provider identifier (openai, megallm, groq, etc.)
            api_key: API key for Bearer token authentication
            base_url: Optional custom base URL (uses provider default if not set)
            model: Model name to use (uses provider default if not set)
            timeout: Request timeout in seconds (defaults to 30.0)
            
        Raises:
            AuthenticationError: If api_key is missing or empty
        """
        self.provider_name = provider_name.lower()
        
        # Validate API key
        if not api_key or not api_key.strip():
            raise AuthenticationError(
                message=f"API key is required for {self.provider_name} provider",
                provider=self.provider_name,
            )
        
        self.api_key = api_key.strip()
        self.base_url = (
            base_url.rstrip("/") if base_url 
            else PROVIDER_BASE_URLS.get(self.provider_name, "https://api.openai.com")
        )
        self.model = model or PROVIDER_DEFAULT_MODELS.get(self.provider_name, "gpt-3.5-turbo")
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        """Return the provider name identifier."""
        return self.provider_name
    
    def _get_headers(self) -> dict[str, str]:
        """
        Get the HTTP headers for API requests.
        
        Returns:
            Dict with Authorization and Content-Type headers
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _get_endpoint(self) -> str:
        """
        Get the chat completions endpoint URL.
        
        Returns:
            Full URL for the chat completions endpoint
        """
        return f"{self.base_url}/v1/chat/completions"

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Send a chat completion request to the OpenAI-compatible API.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Optional model override
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            AIResponse with the model's response
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            QuotaExceededError: If quota is exhausted
            ProviderError: For other API errors
        """
        effective_model = model or self.model
        
        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        logger.debug(
            "Sending request to OpenAI-compatible API",
            provider=self.provider_name,
            model=effective_model,
            base_url=self.base_url,
            message_count=len(messages),
        )
        
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self._get_endpoint(),
                    headers=self._get_headers(),
                    json=payload,
                )
                
                # Handle specific error status codes
                if response.status_code == 401:
                    raise AuthenticationError(
                        message=f"Authentication failed for {self.provider_name}: Invalid API key",
                        provider=self.provider_name,
                        status_code=401,
                    )
                elif response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    raise RateLimitError(
                        message=f"Rate limit exceeded for {self.provider_name}",
                        provider=self.provider_name,
                        retry_after=float(retry_after) if retry_after else None,
                    )
                elif response.status_code == 402 or response.status_code == 403:
                    raise QuotaExceededError(
                        message=f"Quota exceeded or access denied for {self.provider_name}",
                        provider=self.provider_name,
                        status_code=response.status_code,
                    )
                
                response.raise_for_status()
                data = response.json()
                
        except httpx.TimeoutException as e:
            raise ProviderError(
                message=f"{self.provider_name} request timed out after {self.timeout}s",
                provider=self.provider_name,
                retryable=True,
            ) from e
        except httpx.ConnectError as e:
            raise ProviderError(
                message=f"Failed to connect to {self.provider_name} at {self.base_url}",
                provider=self.provider_name,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"{self.provider_name} returned error: {e.response.status_code}",
                provider=self.provider_name,
                status_code=e.response.status_code,
                retryable=e.response.status_code >= 500,
            ) from e
        except (AuthenticationError, RateLimitError, QuotaExceededError):
            # Re-raise our custom exceptions
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse the response
        content = self._extract_content(data)
        usage = self._extract_usage(data)
        
        logger.debug(
            "OpenAI-compatible response received",
            provider=self.provider_name,
            model=effective_model,
            latency_ms=round(latency_ms, 2),
            response_length=len(content),
            usage=usage,
        )
        
        return AIResponse(
            content=content,
            model=data.get("model", effective_model),
            provider=self.provider_name,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    def _extract_content(self, data: dict[str, Any]) -> str:
        """
        Extract the content from an OpenAI-format response.
        
        Args:
            data: The raw API response
            
        Returns:
            The extracted text content
        """
        try:
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
        except (KeyError, IndexError, TypeError):
            logger.warning(
                "Failed to extract content from response",
                provider=self.provider_name,
                response_keys=list(data.keys()) if isinstance(data, dict) else None,
            )
        return ""
    
    def _extract_usage(self, data: dict[str, Any]) -> dict[str, int] | None:
        """
        Extract token usage from an OpenAI-format response.
        
        Args:
            data: The raw API response
            
        Returns:
            Dict with input_tokens and output_tokens, or None if not available
        """
        usage = data.get("usage")
        if usage:
            return {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        return None

    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the OpenAI-compatible provider.
        
        Attempts a minimal API call to verify connectivity and authentication.
        Uses the models endpoint if available, otherwise a minimal chat request.
        
        Returns:
            Dict with health status information including latency metrics
        """
        from .base import HEALTH_CHECK_TIMEOUT, build_health_response
        
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                # Try the models endpoint first (most providers support this)
                models_url = f"{self.base_url}/v1/models"
                response = await client.get(
                    models_url,
                    headers=self._get_headers(),
                )
                
                if response.status_code == 401:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return build_health_response(
                        status="unhealthy",
                        provider=self.provider_name,
                        latency_ms=latency_ms,
                        base_url=self.base_url,
                        model=self.model,
                        error="Authentication failed: Invalid API key",
                    )
                
                response.raise_for_status()
                data = response.json()
                
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract available models if present
            available_models = []
            if "data" in data:
                available_models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            
            model_available = self.model in available_models if available_models else True
            
            return build_health_response(
                status="healthy",
                provider=self.provider_name,
                latency_ms=latency_ms,
                base_url=self.base_url,
                model=self.model,
                model_available=model_available,
                available_models=available_models[:10] if available_models else [],
            )
            
        except httpx.HTTPStatusError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_msg = error_data["error"].get("message", error_msg)
            except Exception:
                pass
            
            logger.warning(
                "OpenAI-compatible health check failed",
                provider=self.provider_name,
                error=error_msg,
                status_code=e.response.status_code,
            )
            return build_health_response(
                status="unhealthy",
                provider=self.provider_name,
                latency_ms=latency_ms,
                base_url=self.base_url,
                model=self.model,
                error=error_msg,
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "OpenAI-compatible health check failed",
                provider=self.provider_name,
                error=str(e),
                base_url=self.base_url,
            )
            return build_health_response(
                status="unhealthy",
                provider=self.provider_name,
                latency_ms=latency_ms,
                base_url=self.base_url,
                model=self.model,
                error=str(e),
            )
    
    @staticmethod
    def build_request_payload(
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Build a request payload for OpenAI-compatible APIs.
        
        This is a static method useful for testing request format correctness.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with the request payload
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        return payload
    
    @staticmethod
    def build_auth_header(api_key: str) -> dict[str, str]:
        """
        Build the authorization header for OpenAI-compatible APIs.
        
        This is a static method useful for testing authentication header correctness.
        
        Args:
            api_key: The API key
            
        Returns:
            Dict with Authorization header
        """
        return {"Authorization": f"Bearer {api_key}"}
