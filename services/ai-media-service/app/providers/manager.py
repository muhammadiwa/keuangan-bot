"""
AI Provider Manager implementation.

This module provides the AIProviderManager class that handles:
- Provider factory based on configuration
- Request routing to appropriate providers
- Fallback mechanism for high availability
- Health monitoring across providers
- Request/response logging with debug mode
- Cost tracking and usage recording (Requirements: 14.1, 14.2)
"""

import re
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from ..config import Settings
from .base import (
    AIProvider,
    AIResponse,
    AuthenticationError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
)
from .anthropic import AnthropicProvider
from .cost_tracker import CostTracker, create_usage_record_from_response
from .gemini import GeminiProvider
from .glm import GLMProvider
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider


# Provider names that use OpenAI-compatible API
OPENAI_COMPATIBLE_PROVIDERS = {
    "openai",
    "megallm",
    "groq",
    "together",
    "deepseek",
    "qwen",
    "kimi",
    "moonshot",
}

# Provider names that map to GLM provider
GLM_PROVIDER_ALIASES = {"glm", "zhipu", "bigmodel"}

# Patterns for redacting sensitive data in logs
SENSITIVE_PATTERNS = [
    (re.compile(r'(api[_-]?key|apikey|authorization|bearer|token|secret|password|credential)["\']?\s*[:=]\s*["\']?([^"\'\s,}\]]+)', re.IGNORECASE), r'\1: [REDACTED]'),
    (re.compile(r'(sk-[a-zA-Z0-9]{20,})', re.IGNORECASE), '[REDACTED_API_KEY]'),
    (re.compile(r'(Bearer\s+)[^\s"\']+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(x-api-key:\s*)[^\s"\']+', re.IGNORECASE), r'\1[REDACTED]'),
]


def redact_sensitive_data(data: Any) -> Any:
    """
    Redact sensitive information from data for safe logging.
    
    Args:
        data: Data to redact (can be dict, list, or string)
        
    Returns:
        Data with sensitive information redacted
    """
    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            # Redact known sensitive keys
            lower_key = key.lower()
            if any(sensitive in lower_key for sensitive in ['api_key', 'apikey', 'secret', 'password', 'token', 'authorization', 'credential']):
                redacted[key] = '[REDACTED]'
            else:
                redacted[key] = redact_sensitive_data(value)
        return redacted
    elif isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]
    elif isinstance(data, str):
        result = data
        for pattern, replacement in SENSITIVE_PATTERNS:
            result = pattern.sub(replacement, result)
        return result
    else:
        return data


class AIProviderManager:
    """
    Manages AI provider selection, fallback, and health monitoring.
    
    This class acts as a facade for all AI providers, handling:
    - Provider instantiation based on configuration
    - Request routing to the appropriate provider
    - Automatic fallback on provider failures
    - Aggregated health status reporting
    
    Attributes:
        primary_provider: The main AI provider for requests
        fallback_provider: Optional backup provider for failures
        config: The application settings
    """

    def __init__(self, config: Settings):
        """
        Initialize the AI Provider Manager.
        
        Args:
            config: Application settings containing provider configuration
        """
        self.config = config
        self._primary_provider: AIProvider | None = None
        self._fallback_provider: AIProvider | None = None
        self._debug_logging = config.ai_debug_logging
        
        # Initialize cost tracker (Requirements: 14.1, 14.2)
        self._cost_tracker = CostTracker(
            backend_api_url=config.backend_api_url,
            enabled=config.ai_cost_tracking_enabled,
        )
        
        # Initialize providers
        self._init_providers()
    
    def _log_request(
        self,
        provider: str,
        model: str | None,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
    ) -> datetime:
        """
        Log an AI request with provider, model, and timestamp.
        
        Args:
            provider: Provider name
            model: Model being used
            messages: Request messages
            temperature: Temperature setting
            max_tokens: Max tokens setting
            
        Returns:
            Request timestamp for latency calculation
        """
        timestamp = datetime.now(timezone.utc)
        
        # Standard request logging (Requirements 13.1)
        logger.info(
            "AI request started",
            provider=provider,
            model=model,
            timestamp=timestamp.isoformat(),
            message_count=len(messages),
        )
        
        # Debug logging with full payloads (Requirements 13.3)
        if self._debug_logging:
            redacted_messages = redact_sensitive_data(messages)
            logger.debug(
                "AI request payload",
                provider=provider,
                model=model,
                messages=redacted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        return timestamp
    
    async def _log_response(
        self,
        provider: str,
        model: str,
        response: "AIResponse",
        request_timestamp: datetime,
        raw_input: str | None = None,
    ) -> None:
        """
        Log an AI response with timing and token usage, and record cost.
        
        Args:
            provider: Provider name
            model: Model used
            response: The AI response
            request_timestamp: When the request was made
            raw_input: The original input text for cost tracking
        """
        response_timestamp = datetime.now(timezone.utc)
        response_time_ms = (response_timestamp - request_timestamp).total_seconds() * 1000
        
        # Standard response logging (Requirements 13.2)
        log_data: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "response_time_ms": round(response_time_ms, 2),
            "latency_ms": round(response.latency_ms, 2),
        }
        
        # Include token usage if available
        if response.usage:
            log_data["input_tokens"] = response.usage.get("input_tokens", 0)
            log_data["output_tokens"] = response.usage.get("output_tokens", 0)
            log_data["total_tokens"] = response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)
        
        logger.info("AI response received", **log_data)
        
        # Debug logging with full response (Requirements 13.3)
        if self._debug_logging:
            redacted_response = redact_sensitive_data({
                "content": response.content[:500] + "..." if len(response.content) > 500 else response.content,
                "model": response.model,
                "provider": response.provider,
                "usage": response.usage,
            })
            logger.debug("AI response payload", **redacted_response)
        
        # Record usage for cost tracking (Requirements 14.1, 14.2)
        try:
            await self._cost_tracker.record_from_response(
                response=response,
                raw_input=raw_input[:1000] if raw_input else None,  # Truncate for storage
                success=True,
            )
        except Exception as e:
            # Don't fail the request if cost tracking fails
            logger.warning("Failed to record usage for cost tracking", error=str(e))
    
    def _log_error(
        self,
        provider: str,
        model: str | None,
        error: Exception,
        request_timestamp: datetime,
    ) -> None:
        """
        Log an AI request error.
        
        Args:
            provider: Provider name
            model: Model being used
            error: The exception that occurred
            request_timestamp: When the request was made
        """
        error_timestamp = datetime.now(timezone.utc)
        elapsed_ms = (error_timestamp - request_timestamp).total_seconds() * 1000
        
        logger.error(
            "AI request failed",
            provider=provider,
            model=model,
            error_type=type(error).__name__,
            error_message=str(error),
            elapsed_ms=round(elapsed_ms, 2),
        )
    
    def _init_providers(self) -> None:
        """Initialize primary and fallback providers based on configuration."""
        # Initialize primary provider
        try:
            self._primary_provider = self._create_provider(
                provider_name=self.config.ai_provider,
                api_key=self.config.ai_api_key,
                base_url=self.config.get_effective_ai_base_url(),
                model=self.config.get_effective_ai_model(),
                timeout=self.config.ai_timeout,
            )
            logger.info(
                "Primary AI provider initialized",
                provider=self.config.ai_provider,
                model=self.config.get_effective_ai_model(),
            )
        except AuthenticationError as e:
            logger.error(
                "Failed to initialize primary provider: authentication error",
                provider=self.config.ai_provider,
                error=str(e),
            )
            self._primary_provider = None
        except Exception as e:
            logger.error(
                "Failed to initialize primary provider",
                provider=self.config.ai_provider,
                error=str(e),
            )
            self._primary_provider = None
        
        # Initialize fallback provider if configured
        if self.config.ai_fallback_provider:
            try:
                self._fallback_provider = self._create_provider(
                    provider_name=self.config.ai_fallback_provider,
                    api_key=self.config.ai_fallback_api_key,
                    base_url=self.config.ai_fallback_base_url,
                    model=self.config.ai_fallback_model,
                    timeout=self.config.ai_timeout,
                )
                logger.info(
                    "Fallback AI provider initialized",
                    provider=self.config.ai_fallback_provider,
                    model=self.config.ai_fallback_model,
                )
            except AuthenticationError as e:
                logger.warning(
                    "Failed to initialize fallback provider: authentication error",
                    provider=self.config.ai_fallback_provider,
                    error=str(e),
                )
                self._fallback_provider = None
            except Exception as e:
                logger.warning(
                    "Failed to initialize fallback provider",
                    provider=self.config.ai_fallback_provider,
                    error=str(e),
                )
                self._fallback_provider = None
    
    @staticmethod
    def _create_provider(
        provider_name: str,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        timeout: float = 30.0,
    ) -> AIProvider:
        """
        Factory method to create an AI provider instance.
        
        Args:
            provider_name: Name of the provider (ollama, openai, anthropic, etc.)
            api_key: API key for authentication (if required)
            base_url: Custom base URL (optional)
            model: Model name to use (optional)
            timeout: Request timeout in seconds
            
        Returns:
            An AIProvider instance
            
        Raises:
            AuthenticationError: If API key is required but missing
            ValueError: If provider name is unknown
        """
        provider_name = provider_name.lower()
        
        # Ollama provider (no API key required)
        if provider_name == "ollama":
            return OllamaProvider(
                base_url=base_url,
                model=model,
                timeout=timeout,
            )
        
        # OpenAI-compatible providers
        if provider_name in OPENAI_COMPATIBLE_PROVIDERS:
            if not api_key:
                raise AuthenticationError(
                    message=f"API key is required for {provider_name} provider",
                    provider=provider_name,
                )
            return OpenAICompatibleProvider(
                provider_name=provider_name,
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
            )
        
        # GLM provider (Zhipu AI)
        if provider_name in GLM_PROVIDER_ALIASES:
            if not api_key:
                raise AuthenticationError(
                    message=f"API key is required for {provider_name} provider",
                    provider=provider_name,
                )
            return GLMProvider(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
            )
        
        # Anthropic provider
        if provider_name == "anthropic":
            if not api_key:
                raise AuthenticationError(
                    message="API key is required for Anthropic provider",
                    provider=provider_name,
                )
            return AnthropicProvider(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
            )
        
        # Gemini provider
        if provider_name == "gemini":
            if not api_key:
                raise AuthenticationError(
                    message="API key is required for Gemini provider",
                    provider=provider_name,
                )
            return GeminiProvider(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
            )
        
        raise ValueError(f"Unknown provider: {provider_name}")
    
    @property
    def primary_provider(self) -> AIProvider | None:
        """Get the primary AI provider."""
        return self._primary_provider
    
    @property
    def fallback_provider(self) -> AIProvider | None:
        """Get the fallback AI provider."""
        return self._fallback_provider
    
    @property
    def has_primary(self) -> bool:
        """Check if a primary provider is available."""
        return self._primary_provider is not None
    
    @property
    def has_fallback(self) -> bool:
        """Check if a fallback provider is available."""
        return self._fallback_provider is not None

    
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        use_fallback: bool = True,
    ) -> AIResponse:
        """
        Send a chat completion request with automatic fallback.
        
        Routes the request to the primary provider. If the primary fails
        and a fallback is configured, retries with the fallback provider.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Optional model override
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            use_fallback: Whether to use fallback on primary failure
            
        Returns:
            AIResponse with the model's response
            
        Raises:
            ProviderError: If all providers fail
        """
        primary_error: Exception | None = None
        fallback_error: Exception | None = None
        
        # Try primary provider
        if self._primary_provider:
            primary_model = model or self.config.get_effective_ai_model()
            request_timestamp = self._log_request(
                provider=self._primary_provider.name,
                model=primary_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Extract raw input for cost tracking
            raw_input = None
            for msg in messages:
                if msg.get("role") == "user":
                    raw_input = msg.get("content", "")
                    break
            
            try:
                response = await self._primary_provider.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                await self._log_response(
                    provider=self._primary_provider.name,
                    model=response.model,
                    response=response,
                    request_timestamp=request_timestamp,
                    raw_input=raw_input,
                )
                return response
            except (ProviderError, AuthenticationError, RateLimitError, QuotaExceededError) as e:
                primary_error = e
                self._log_error(
                    provider=self._primary_provider.name,
                    model=primary_model,
                    error=e,
                    request_timestamp=request_timestamp,
                )
                logger.warning(
                    "Primary provider failed",
                    provider=self._primary_provider.name,
                    error=str(e),
                    retryable=getattr(e, "retryable", False),
                )
        else:
            primary_error = ProviderError(
                message="No primary provider configured",
                provider="none",
            )
        
        # Try fallback provider if enabled and available
        if use_fallback and self._fallback_provider:
            fallback_model = model or self.config.ai_fallback_model
            fallback_timestamp = self._log_request(
                provider=self._fallback_provider.name,
                model=fallback_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Extract raw input for cost tracking
            fallback_raw_input = None
            for msg in messages:
                if msg.get("role") == "user":
                    fallback_raw_input = msg.get("content", "")
                    break
            
            try:
                logger.info(
                    "Attempting fallback provider",
                    fallback_provider=self._fallback_provider.name,
                    primary_error=str(primary_error),
                )
                response = await self._fallback_provider.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                await self._log_response(
                    provider=self._fallback_provider.name,
                    model=response.model,
                    response=response,
                    request_timestamp=fallback_timestamp,
                    raw_input=fallback_raw_input,
                )
                logger.info(
                    "Fallback provider succeeded",
                    provider=self._fallback_provider.name,
                )
                return response
            except (ProviderError, AuthenticationError, RateLimitError, QuotaExceededError) as e:
                fallback_error = e
                self._log_error(
                    provider=self._fallback_provider.name,
                    model=fallback_model,
                    error=e,
                    request_timestamp=fallback_timestamp,
                )
                logger.error(
                    "Fallback provider also failed",
                    provider=self._fallback_provider.name,
                    error=str(e),
                )
        
        # All providers failed - raise error with context
        if fallback_error:
            raise ProviderError(
                message=f"All providers failed. Primary: {primary_error}. Fallback: {fallback_error}",
                provider="manager",
                retryable=False,
            )
        
        # No fallback available or disabled, raise the primary error
        if primary_error:
            raise primary_error
        
        raise ProviderError(
            message="No AI providers available",
            provider="manager",
        )
    
    async def chat_completion_with_heuristic_fallback(
        self,
        messages: list[dict[str, str]],
        heuristic_fn: callable,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> tuple[AIResponse | None, Any]:
        """
        Send a chat completion request with heuristic fallback on total failure.
        
        This method attempts AI providers first, and if all fail, falls back
        to a heuristic function as a last resort.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            heuristic_fn: Callable that takes the last user message and returns a result
            model: Optional model override
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (AIResponse or None, heuristic_result or None)
            - If AI succeeds: (AIResponse, None)
            - If heuristic used: (None, heuristic_result)
        """
        try:
            response = await self.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                use_fallback=True,
            )
            return response, None
        except ProviderError as e:
            logger.warning(
                "All AI providers failed, using heuristic fallback",
                error=str(e),
            )
            
            # Extract the last user message for heuristic processing
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            heuristic_timestamp = datetime.now(timezone.utc)
            try:
                heuristic_result = heuristic_fn(user_message)
                elapsed_ms = (datetime.now(timezone.utc) - heuristic_timestamp).total_seconds() * 1000
                logger.info(
                    "Heuristic fallback succeeded",
                    provider="heuristic",
                    input_length=len(user_message),
                    response_time_ms=round(elapsed_ms, 2),
                )
                return None, heuristic_result
            except Exception as heuristic_error:
                elapsed_ms = (datetime.now(timezone.utc) - heuristic_timestamp).total_seconds() * 1000
                logger.error(
                    "Heuristic fallback also failed",
                    provider="heuristic",
                    error=str(heuristic_error),
                    elapsed_ms=round(elapsed_ms, 2),
                )
                raise ProviderError(
                    message=f"All providers and heuristic failed. AI error: {e}. Heuristic error: {heuristic_error}",
                    provider="manager",
                    retryable=False,
                ) from heuristic_error
    
    async def get_health_status(self) -> dict[str, Any]:
        """
        Get aggregated health status of all configured providers.
        
        The overall status is determined as follows:
        - 'healthy': Primary provider is healthy with acceptable latency
        - 'degraded': Primary has latency warning, or primary unhealthy but fallback healthy
        - 'unhealthy': All providers are unhealthy or unavailable
        
        Returns:
            Dict with overall status, individual provider statuses, and latency metrics
        """
        from .base import LATENCY_THRESHOLD_MS
        
        result: dict[str, Any] = {
            "status": "healthy",
            "providers": {},
            "latency_threshold_ms": LATENCY_THRESHOLD_MS,
        }
        
        # Check primary provider
        if self._primary_provider:
            primary_health = await self._primary_provider.health_check()
            result["providers"]["primary"] = {
                "name": self._primary_provider.name,
                **primary_health,
            }
            
            # Check for latency warning or unhealthy status
            if primary_health.get("status") != "healthy":
                result["status"] = "degraded"
            elif primary_health.get("latency_warning"):
                result["status"] = "degraded"
        else:
            result["providers"]["primary"] = {
                "name": self.config.ai_provider,
                "status": "unhealthy",
                "error": "Provider not initialized",
                "latency_ms": 0,
            }
            result["status"] = "degraded"
        
        # Check fallback provider
        if self._fallback_provider:
            fallback_health = await self._fallback_provider.health_check()
            result["providers"]["fallback"] = {
                "name": self._fallback_provider.name,
                **fallback_health,
            }
        elif self.config.ai_fallback_provider:
            result["providers"]["fallback"] = {
                "name": self.config.ai_fallback_provider,
                "status": "unhealthy",
                "error": "Provider not initialized",
                "latency_ms": 0,
            }
        
        # Determine final status based on provider health
        primary_status = result["providers"].get("primary", {}).get("status")
        fallback_status = result["providers"].get("fallback", {}).get("status")
        primary_latency_warning = result["providers"].get("primary", {}).get("latency_warning", False)
        
        if primary_status != "healthy":
            if fallback_status == "healthy":
                result["status"] = "degraded"
            else:
                result["status"] = "unhealthy"
        elif primary_latency_warning:
            # Primary is healthy but slow
            result["status"] = "degraded"
        
        # Add summary metrics
        result["summary"] = self._build_health_summary(result["providers"])
        
        return result
    
    def _build_health_summary(self, providers: dict[str, Any]) -> dict[str, Any]:
        """
        Build a summary of health metrics from provider statuses.
        
        Args:
            providers: Dict of provider health statuses
            
        Returns:
            Dict with summary metrics
        """
        summary: dict[str, Any] = {
            "total_providers": 0,
            "healthy_providers": 0,
            "unhealthy_providers": 0,
            "providers_with_latency_warning": 0,
        }
        
        for role, health in providers.items():
            if health:
                summary["total_providers"] += 1
                status = health.get("status", "unknown")
                
                if status == "healthy":
                    summary["healthy_providers"] += 1
                elif status == "unhealthy":
                    summary["unhealthy_providers"] += 1
                
                if health.get("latency_warning"):
                    summary["providers_with_latency_warning"] += 1
        
        return summary
    
    def get_provider_info(self) -> dict[str, Any]:
        """
        Get information about configured providers.
        
        Returns:
            Dict with provider configuration details
        """
        info: dict[str, Any] = {
            "primary": None,
            "fallback": None,
        }
        
        if self._primary_provider:
            info["primary"] = {
                "name": self._primary_provider.name,
                "available": True,
            }
        else:
            info["primary"] = {
                "name": self.config.ai_provider,
                "available": False,
            }
        
        if self._fallback_provider:
            info["fallback"] = {
                "name": self._fallback_provider.name,
                "available": True,
            }
        elif self.config.ai_fallback_provider:
            info["fallback"] = {
                "name": self.config.ai_fallback_provider,
                "available": False,
            }
        
        return info
