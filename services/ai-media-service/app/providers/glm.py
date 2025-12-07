"""
GLM (Zhipu AI / BigModel) Provider implementation.

This module provides integration with Zhipu AI's BigModel API for GLM models.
GLM uses JWT token authentication generated from the API key.

Supported models:
- glm-4
- glm-4-flash
- glm-3-turbo
"""

import hashlib
import hmac
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


class GLMProvider(AIProvider):
    """
    AI Provider implementation for Zhipu AI's GLM models.
    
    GLM (General Language Model) is developed by Zhipu AI and accessed via
    the BigModel API (open.bigmodel.cn). It uses JWT token authentication
    where the API key is in the format {id}.{secret}.
    
    Attributes:
        api_key: The API key in format {id}.{secret}
        base_url: The API base URL (default: https://open.bigmodel.cn/api/paas)
        model: The model to use (e.g., glm-4, glm-4-flash, glm-3-turbo)
        timeout: Request timeout in seconds
    """
    
    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas"
    DEFAULT_MODEL = "glm-4-flash"
    
    # Token expiration time in seconds (default: 3600 = 1 hour)
    TOKEN_EXPIRATION = 3600
    
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the GLM provider.
        
        Args:
            api_key: API key in format {id}.{secret}
            base_url: Optional custom base URL (defaults to BigModel API)
            model: Model name to use (defaults to glm-4-flash)
            timeout: Request timeout in seconds (defaults to 30.0)
            
        Raises:
            AuthenticationError: If api_key is missing or invalid format
        """
        # Validate API key
        if not api_key or not api_key.strip():
            raise AuthenticationError(
                message="API key is required for GLM provider",
                provider="glm",
            )
        
        api_key = api_key.strip()
        
        # Validate API key format (should be {id}.{secret})
        if "." not in api_key:
            raise AuthenticationError(
                message="GLM API key must be in format {id}.{secret}",
                provider="glm",
            )
        
        parts = api_key.split(".", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise AuthenticationError(
                message="GLM API key must be in format {id}.{secret}",
                provider="glm",
            )
        
        self.api_key = api_key
        self._api_key_id = parts[0]
        self._api_key_secret = parts[1]
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        """Return the provider name identifier."""
        return "glm"
    
    def generate_jwt_token(self, expiration_seconds: int | None = None) -> str:
        """
        Generate a JWT token for authentication with Zhipu AI API.
        
        The JWT token is generated using HS256 algorithm with the following structure:
        - Header: {"alg": "HS256", "sign_type": "SIGN"}
        - Payload: {"api_key": id, "exp": expiration_timestamp, "timestamp": current_timestamp}
        
        Args:
            expiration_seconds: Token expiration time in seconds (defaults to TOKEN_EXPIRATION)
            
        Returns:
            JWT token string
        """
        import base64
        import json
        
        exp_seconds = expiration_seconds or self.TOKEN_EXPIRATION
        
        # Current timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        
        # Expiration timestamp in milliseconds
        exp_ms = timestamp_ms + (exp_seconds * 1000)
        
        # JWT Header
        header = {
            "alg": "HS256",
            "sign_type": "SIGN"
        }
        
        # JWT Payload
        payload = {
            "api_key": self._api_key_id,
            "exp": exp_ms,
            "timestamp": timestamp_ms
        }
        
        # Base64url encode header and payload
        header_b64 = self._base64url_encode(json.dumps(header, separators=(",", ":")))
        payload_b64 = self._base64url_encode(json.dumps(payload, separators=(",", ":")))
        
        # Create signature
        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self._api_key_secret.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256
        ).digest()
        signature_b64 = self._base64url_encode_bytes(signature)
        
        # Combine to form JWT
        return f"{header_b64}.{payload_b64}.{signature_b64}"
    
    @staticmethod
    def _base64url_encode(data: str) -> str:
        """Base64url encode a string."""
        import base64
        return base64.urlsafe_b64encode(data.encode("utf-8")).rstrip(b"=").decode("utf-8")
    
    @staticmethod
    def _base64url_encode_bytes(data: bytes) -> str:
        """Base64url encode bytes."""
        import base64
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")
    
    def _get_headers(self) -> dict[str, str]:
        """
        Get the HTTP headers for API requests.
        
        Returns:
            Dict with Authorization and Content-Type headers
        """
        token = self.generate_jwt_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
    
    def _get_endpoint(self) -> str:
        """
        Get the chat completions endpoint URL.
        
        Returns:
            Full URL for the chat completions endpoint
        """
        return f"{self.base_url}/v4/chat/completions"

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Send a chat completion request to the GLM API.
        
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
            "Sending request to GLM API",
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
                        message="Authentication failed for GLM: Invalid API key or JWT token",
                        provider=self.name,
                        status_code=401,
                    )
                elif response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    raise RateLimitError(
                        message="Rate limit exceeded for GLM",
                        provider=self.name,
                        retry_after=float(retry_after) if retry_after else None,
                    )
                elif response.status_code == 402 or response.status_code == 403:
                    raise QuotaExceededError(
                        message="Quota exceeded or access denied for GLM",
                        provider=self.name,
                        status_code=response.status_code,
                    )
                
                response.raise_for_status()
                data = response.json()
                
        except httpx.TimeoutException as e:
            raise ProviderError(
                message=f"GLM request timed out after {self.timeout}s",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.ConnectError as e:
            raise ProviderError(
                message=f"Failed to connect to GLM at {self.base_url}",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"GLM returned error: {e.response.status_code}",
                provider=self.name,
                status_code=e.response.status_code,
                retryable=e.response.status_code >= 500,
            ) from e
        except (AuthenticationError, RateLimitError, QuotaExceededError):
            # Re-raise our custom exceptions
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse the response (OpenAI-compatible format)
        content = self._extract_content(data)
        usage = self._extract_usage(data)
        
        logger.debug(
            "GLM response received",
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
    
    def _extract_content(self, data: dict[str, Any]) -> str:
        """
        Extract the content from a GLM response.
        
        GLM uses OpenAI-compatible response format.
        
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
                "Failed to extract content from GLM response",
                response_keys=list(data.keys()) if isinstance(data, dict) else None,
            )
        return ""
    
    def _extract_usage(self, data: dict[str, Any]) -> dict[str, int] | None:
        """
        Extract token usage from a GLM response.
        
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
        Check the health status of the GLM provider.
        
        Attempts a minimal API call to verify connectivity and authentication.
        
        Returns:
            Dict with health status information including latency metrics
        """
        from .base import HEALTH_CHECK_TIMEOUT, build_health_response
        
        start_time = time.perf_counter()
        
        try:
            # GLM doesn't have a models endpoint, so we do a minimal chat request
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
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
                        error="Authentication failed: Invalid API key or JWT token",
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
                "GLM health check failed",
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
                "GLM health check failed",
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
    def generate_jwt_from_api_key(
        api_key: str,
        expiration_seconds: int = 3600,
    ) -> str:
        """
        Generate a JWT token from an API key.
        
        This is a static method useful for testing JWT generation correctness.
        
        Args:
            api_key: API key in format {id}.{secret}
            expiration_seconds: Token expiration time in seconds
            
        Returns:
            JWT token string
            
        Raises:
            ValueError: If api_key format is invalid
        """
        import base64
        import json
        
        if not api_key or "." not in api_key:
            raise ValueError("API key must be in format {id}.{secret}")
        
        parts = api_key.split(".", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("API key must be in format {id}.{secret}")
        
        api_key_id = parts[0]
        api_key_secret = parts[1]
        
        # Current timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        
        # Expiration timestamp in milliseconds
        exp_ms = timestamp_ms + (expiration_seconds * 1000)
        
        # JWT Header
        header = {
            "alg": "HS256",
            "sign_type": "SIGN"
        }
        
        # JWT Payload
        payload = {
            "api_key": api_key_id,
            "exp": exp_ms,
            "timestamp": timestamp_ms
        }
        
        # Base64url encode helper
        def b64url_encode(data: str) -> str:
            return base64.urlsafe_b64encode(data.encode("utf-8")).rstrip(b"=").decode("utf-8")
        
        def b64url_encode_bytes(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")
        
        # Base64url encode header and payload
        header_b64 = b64url_encode(json.dumps(header, separators=(",", ":")))
        payload_b64 = b64url_encode(json.dumps(payload, separators=(",", ":")))
        
        # Create signature
        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            api_key_secret.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256
        ).digest()
        signature_b64 = b64url_encode_bytes(signature)
        
        # Combine to form JWT
        return f"{header_b64}.{payload_b64}.{signature_b64}"
    
    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """
        Validate that an API key is in the correct format for GLM.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            True if the format is valid, False otherwise
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        api_key = api_key.strip()
        if "." not in api_key:
            return False
        
        parts = api_key.split(".", 1)
        return len(parts) == 2 and bool(parts[0]) and bool(parts[1])
