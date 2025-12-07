"""
Google Gemini Provider implementation.

This module provides integration with Google's Generative AI API (Gemini).
Gemini uses a unique API format with API key in query parameters and
candidates-based response format.

Supported models:
- gemini-1.5-pro
- gemini-1.5-flash
- gemini-1.0-pro
- gemini-pro
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


class GeminiProvider(AIProvider):
    """
    AI Provider implementation for Google's Gemini models.
    
    Gemini uses a unique API format with:
    - POST /v1beta/models/{model}:generateContent endpoint
    - API key in query parameters (not headers)
    - Candidates-based response format
    
    Attributes:
        api_key: The Google API key
        base_url: The API base URL (default: https://generativelanguage.googleapis.com)
        model: The model to use (e.g., gemini-1.5-flash)
        timeout: Request timeout in seconds
    """
    
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Google API key
            base_url: Optional custom base URL (defaults to generativelanguage.googleapis.com)
            model: Model name to use (defaults to gemini-1.5-flash)
            timeout: Request timeout in seconds (defaults to 30.0)
            
        Raises:
            AuthenticationError: If api_key is missing or empty
        """
        # Validate API key
        if not api_key or not api_key.strip():
            raise AuthenticationError(
                message="API key is required for Gemini provider",
                provider="gemini",
            )
        
        self.api_key = api_key.strip()
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        """Return the provider name identifier."""
        return "gemini"
    
    def _get_endpoint(self, model: str) -> str:
        """
        Get the generateContent endpoint URL with API key.
        
        Gemini uses API key in query parameters, not headers.
        
        Args:
            model: The model name to use
            
        Returns:
            Full URL for the generateContent endpoint with API key
        """
        return f"{self.base_url}/v1beta/models/{model}:generateContent?key={self.api_key}"
    
    def _get_headers(self) -> dict[str, str]:
        """
        Get the HTTP headers for API requests.
        
        Gemini doesn't use Authorization header, just Content-Type.
        
        Returns:
            Dict with Content-Type header
        """
        return {
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Send a chat completion request to the Gemini API.
        
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
        
        # Convert messages to Gemini format
        gemini_contents, system_instruction = self._convert_messages(messages)
        
        payload: dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        
        # Add max tokens if specified
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        
        # Add system instruction if present
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        logger.debug(
            "Sending request to Gemini API",
            model=effective_model,
            base_url=self.base_url,
            content_count=len(gemini_contents),
        )
        
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self._get_endpoint(effective_model),
                    headers=self._get_headers(),
                    json=payload,
                )
                
                # Handle specific error status codes
                if response.status_code == 400:
                    error_data = response.json()
                    error_msg = self._extract_error_message(error_data)
                    if "API_KEY" in error_msg.upper() or "API key" in error_msg:
                        raise AuthenticationError(
                            message=f"Authentication failed for Gemini: {error_msg}",
                            provider=self.name,
                            status_code=400,
                        )
                    raise ProviderError(
                        message=f"Gemini bad request: {error_msg}",
                        provider=self.name,
                        status_code=400,
                    )
                elif response.status_code == 401 or response.status_code == 403:
                    raise AuthenticationError(
                        message="Authentication failed for Gemini: Invalid API key",
                        provider=self.name,
                        status_code=response.status_code,
                    )
                elif response.status_code == 429:
                    raise RateLimitError(
                        message="Rate limit exceeded for Gemini",
                        provider=self.name,
                    )
                
                response.raise_for_status()
                data = response.json()
                
        except httpx.TimeoutException as e:
            raise ProviderError(
                message=f"Gemini request timed out after {self.timeout}s",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.ConnectError as e:
            raise ProviderError(
                message=f"Failed to connect to Gemini at {self.base_url}",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"Gemini returned error: {e.response.status_code}",
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
            "Gemini response received",
            model=effective_model,
            latency_ms=round(latency_ms, 2),
            response_length=len(content),
            usage=usage,
        )
        
        return AIResponse(
            content=content,
            model=effective_model,
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=data,
        )

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Convert OpenAI-style messages to Gemini format.
        
        Gemini uses:
        - 'contents' array with 'role' and 'parts' structure
        - 'user' and 'model' roles (not 'assistant')
        - System messages as separate 'systemInstruction'
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Tuple of (gemini_contents, system_instruction)
        """
        system_instruction = None
        gemini_contents = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Collect system messages
                if system_instruction:
                    system_instruction += "\n" + content
                else:
                    system_instruction = content
            else:
                # Map roles: Gemini uses 'user' and 'model'
                gemini_role = "model" if role == "assistant" else "user"
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}],
                })
        
        return gemini_contents, system_instruction
    
    def _extract_content(self, data: dict[str, Any]) -> str:
        """
        Extract the content from a Gemini response.
        
        Gemini returns content in candidates[].content.parts[].text format.
        
        Args:
            data: The raw API response
            
        Returns:
            The extracted text content
        """
        return self.parse_candidates(data.get("candidates", []))
    
    def _extract_usage(self, data: dict[str, Any]) -> dict[str, int] | None:
        """
        Extract token usage from a Gemini response.
        
        Args:
            data: The raw API response
            
        Returns:
            Dict with input_tokens and output_tokens, or None if not available
        """
        usage_metadata = data.get("usageMetadata")
        if usage_metadata:
            return {
                "input_tokens": usage_metadata.get("promptTokenCount", 0),
                "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
            }
        return None
    
    def _extract_error_message(self, data: dict[str, Any]) -> str:
        """
        Extract error message from Gemini error response.
        
        Args:
            data: The error response data
            
        Returns:
            Error message string
        """
        try:
            error = data.get("error", {})
            return error.get("message", str(data))
        except Exception:
            return str(data)

    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the Gemini provider.
        
        Attempts a minimal API call to verify connectivity and authentication.
        
        Returns:
            Dict with health status information including latency metrics
        """
        from .base import HEALTH_CHECK_TIMEOUT, build_health_response
        
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                # Use a minimal request to check health
                response = await client.post(
                    self._get_endpoint(self.model),
                    headers=self._get_headers(),
                    json={
                        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                        "generationConfig": {"maxOutputTokens": 5},
                    },
                )
                
                if response.status_code in (400, 401, 403):
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    error_msg = "Authentication failed: Invalid API key"
                    try:
                        error_data = response.json()
                        error_msg = self._extract_error_message(error_data)
                    except Exception:
                        pass
                    return build_health_response(
                        status="unhealthy",
                        provider=self.name,
                        latency_ms=latency_ms,
                        base_url=self.base_url,
                        model=self.model,
                        error=error_msg,
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
                error_msg = self._extract_error_message(error_data)
            except Exception:
                pass
            
            logger.warning(
                "Gemini health check failed",
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
                "Gemini health check failed",
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
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Build a request payload for Gemini API.
        
        This is a static method useful for testing request format correctness.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with the request payload in Gemini format
        """
        system_instruction = None
        gemini_contents = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                if system_instruction:
                    system_instruction += "\n" + content
                else:
                    system_instruction = content
            else:
                gemini_role = "model" if role == "assistant" else "user"
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}],
                })
        
        payload: dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        return payload
    
    @staticmethod
    def build_endpoint_url(base_url: str, model: str, api_key: str) -> str:
        """
        Build the endpoint URL with API key in query params.
        
        This is a static method useful for testing endpoint format correctness.
        Gemini uses API key in query parameters, not headers.
        
        Args:
            base_url: The base URL for the API
            model: The model name
            api_key: The API key
            
        Returns:
            Full endpoint URL with API key
        """
        base_url = base_url.rstrip("/")
        return f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"
    
    @staticmethod
    def parse_candidates(candidates: list[dict[str, Any]]) -> str:
        """
        Parse Gemini candidates to extract text content.
        
        This is a static method useful for testing response parsing.
        
        Args:
            candidates: List of candidate dicts from Gemini response
            
        Returns:
            Extracted text content from the first candidate
        """
        try:
            if not candidates:
                return ""
            
            # Get the first candidate
            first_candidate = candidates[0]
            content = first_candidate.get("content", {})
            parts = content.get("parts", [])
            
            text_parts = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
            
            return "".join(text_parts)
        except (KeyError, TypeError, IndexError):
            return ""
