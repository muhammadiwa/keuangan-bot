"""
Anthropic Claude Provider implementation.

This module provides integration with Anthropic's Claude API for high-quality NLU.
Anthropic uses a different API format than OpenAI with x-api-key authentication
and content blocks response format.

Supported models:
- claude-3-5-sonnet-20241022
- claude-3-5-haiku-20241022
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307
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


class AnthropicProvider(AIProvider):
    """
    AI Provider implementation for Anthropic's Claude models.
    
    Anthropic uses a unique API format with:
    - POST /v1/messages endpoint
    - x-api-key header for authentication
    - Content blocks response format
    
    Attributes:
        api_key: The Anthropic API key
        base_url: The API base URL (default: https://api.anthropic.com)
        model: The model to use (e.g., claude-3-5-sonnet-20241022)
        timeout: Request timeout in seconds
    """
    
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    ANTHROPIC_VERSION = "2023-06-01"
    
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            base_url: Optional custom base URL (defaults to api.anthropic.com)
            model: Model name to use (defaults to claude-3-5-sonnet-20241022)
            timeout: Request timeout in seconds (defaults to 30.0)
            
        Raises:
            AuthenticationError: If api_key is missing or empty
        """
        # Validate API key
        if not api_key or not api_key.strip():
            raise AuthenticationError(
                message="API key is required for Anthropic provider",
                provider="anthropic",
            )
        
        self.api_key = api_key.strip()
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        """Return the provider name identifier."""
        return "anthropic"
    
    def _get_headers(self) -> dict[str, str]:
        """
        Get the HTTP headers for API requests.
        
        Anthropic uses x-api-key header instead of Bearer token.
        
        Returns:
            Dict with x-api-key, anthropic-version, and Content-Type headers
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
    
    def _get_endpoint(self) -> str:
        """
        Get the messages endpoint URL.
        
        Returns:
            Full URL for the messages endpoint
        """
        return f"{self.base_url}/v1/messages"


    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Send a chat completion request to the Anthropic API.
        
        Anthropic requires messages to start with a user message and uses
        a different request/response format than OpenAI.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Optional model override
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response (defaults to 1024)
            
        Returns:
            AIResponse with the model's response
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            QuotaExceededError: If quota is exhausted
            ProviderError: For other API errors
        """
        effective_model = model or self.model
        effective_max_tokens = max_tokens or 1024
        
        # Convert messages to Anthropic format
        anthropic_messages, system_prompt = self._convert_messages(messages)
        
        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": anthropic_messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }
        
        # Add system prompt if present
        if system_prompt:
            payload["system"] = system_prompt
        
        logger.debug(
            "Sending request to Anthropic API",
            model=effective_model,
            base_url=self.base_url,
            message_count=len(anthropic_messages),
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
                        message="Authentication failed for Anthropic: Invalid API key",
                        provider=self.name,
                        status_code=401,
                    )
                elif response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    raise RateLimitError(
                        message="Rate limit exceeded for Anthropic",
                        provider=self.name,
                        retry_after=float(retry_after) if retry_after else None,
                    )
                elif response.status_code == 402 or response.status_code == 403:
                    raise QuotaExceededError(
                        message="Quota exceeded or access denied for Anthropic",
                        provider=self.name,
                        status_code=response.status_code,
                    )
                
                response.raise_for_status()
                data = response.json()
                
        except httpx.TimeoutException as e:
            raise ProviderError(
                message=f"Anthropic request timed out after {self.timeout}s",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.ConnectError as e:
            raise ProviderError(
                message=f"Failed to connect to Anthropic at {self.base_url}",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"Anthropic returned error: {e.response.status_code}",
                provider=self.name,
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
            "Anthropic response received",
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
    
    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], str | None]:
        """
        Convert OpenAI-style messages to Anthropic format.
        
        Anthropic requires:
        - System messages to be passed separately as 'system' parameter
        - Messages to alternate between user and assistant
        - First message must be from user
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Tuple of (converted_messages, system_prompt)
        """
        system_prompt = None
        converted_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Collect system messages
                if system_prompt:
                    system_prompt += "\n" + content
                else:
                    system_prompt = content
            else:
                # Map roles (Anthropic uses 'user' and 'assistant')
                anthropic_role = "assistant" if role == "assistant" else "user"
                converted_messages.append({
                    "role": anthropic_role,
                    "content": content,
                })
        
        # Ensure first message is from user (Anthropic requirement)
        if converted_messages and converted_messages[0]["role"] != "user":
            # Prepend an empty user message if needed
            converted_messages.insert(0, {"role": "user", "content": "Hello"})
        
        return converted_messages, system_prompt

    
    def _extract_content(self, data: dict[str, Any]) -> str:
        """
        Extract the content from an Anthropic response.
        
        Anthropic returns content as an array of content blocks, each with
        a 'type' and 'text' field for text content.
        
        Args:
            data: The raw API response
            
        Returns:
            The extracted text content
        """
        try:
            content_blocks = data.get("content", [])
            text_parts = []
            
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            
            return "".join(text_parts)
        except (KeyError, TypeError):
            logger.warning(
                "Failed to extract content from Anthropic response",
                response_keys=list(data.keys()) if isinstance(data, dict) else None,
            )
        return ""
    
    def _extract_usage(self, data: dict[str, Any]) -> dict[str, int] | None:
        """
        Extract token usage from an Anthropic response.
        
        Args:
            data: The raw API response
            
        Returns:
            Dict with input_tokens and output_tokens, or None if not available
        """
        usage = data.get("usage")
        if usage:
            return {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
        return None

    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the Anthropic provider.
        
        Attempts a minimal API call to verify connectivity and authentication.
        
        Returns:
            Dict with health status information including latency metrics
        """
        from .base import HEALTH_CHECK_TIMEOUT, build_health_response
        
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                # Anthropic doesn't have a models endpoint, so we do a minimal chat request
                response = await client.post(
                    self._get_endpoint(),
                    headers=self._get_headers(),
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 5,
                    },
                )
                
                if response.status_code == 401:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return build_health_response(
                        status="unhealthy",
                        provider=self.name,
                        latency_ms=latency_ms,
                        base_url=self.base_url,
                        model=self.model,
                        error="Authentication failed: Invalid API key",
                    )
                
                response.raise_for_status()
                
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return build_health_response(
                status="healthy",
                provider=self.name,
                latency_ms=latency_ms,
                base_url=self.base_url,
                model=self.model,
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
                "Anthropic health check failed",
                error=error_msg,
                status_code=e.response.status_code,
            )
            return build_health_response(
                status="unhealthy",
                provider=self.name,
                latency_ms=latency_ms,
                base_url=self.base_url,
                model=self.model,
                error=error_msg,
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Anthropic health check failed",
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
    
    @staticmethod
    def build_request_payload(
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """
        Build a request payload for Anthropic API.
        
        This is a static method useful for testing request format correctness.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with the request payload in Anthropic format
        """
        system_prompt = None
        converted_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                if system_prompt:
                    system_prompt += "\n" + content
                else:
                    system_prompt = content
            else:
                anthropic_role = "assistant" if role == "assistant" else "user"
                converted_messages.append({
                    "role": anthropic_role,
                    "content": content,
                })
        
        # Ensure first message is from user
        if converted_messages and converted_messages[0]["role"] != "user":
            converted_messages.insert(0, {"role": "user", "content": "Hello"})
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        return payload
    
    @staticmethod
    def build_auth_header(api_key: str) -> dict[str, str]:
        """
        Build the authentication header for Anthropic API.
        
        This is a static method useful for testing authentication header correctness.
        Anthropic uses x-api-key header instead of Bearer token.
        
        Args:
            api_key: The API key
            
        Returns:
            Dict with x-api-key header
        """
        return {"x-api-key": api_key}
    
    @staticmethod
    def parse_content_blocks(content_blocks: list[dict[str, Any]]) -> str:
        """
        Parse Anthropic content blocks to extract text.
        
        This is a static method useful for testing response parsing.
        
        Args:
            content_blocks: List of content block dicts from Anthropic response
            
        Returns:
            Extracted text content
        """
        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)
