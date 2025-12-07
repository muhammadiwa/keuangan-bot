"""
AI Cost Tracking Service.

This module provides functionality to record AI provider usage and costs
to the backend-api's ai_audit table for monitoring and cost analysis.

Requirements: 14.1, 14.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from .base import AIResponse


# Cost per 1K tokens for various providers (in USD)
# These are approximate costs and should be updated as pricing changes
PROVIDER_COSTS: dict[str, dict[str, float]] = {
    "openai": {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "default": {"input": 0.001, "output": 0.002},
    },
    "anthropic": {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "default": {"input": 0.003, "output": 0.015},
    },
    "gemini": {
        "gemini-pro": {"input": 0.00025, "output": 0.0005},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "default": {"input": 0.00025, "output": 0.0005},
    },
    "groq": {
        "llama-3.1-70b": {"input": 0.00059, "output": 0.00079},
        "llama-3.1-8b": {"input": 0.00005, "output": 0.00008},
        "mixtral-8x7b": {"input": 0.00024, "output": 0.00024},
        "default": {"input": 0.0003, "output": 0.0003},
    },
    "deepseek": {
        "deepseek-chat": {"input": 0.00014, "output": 0.00028},
        "deepseek-coder": {"input": 0.00014, "output": 0.00028},
        "default": {"input": 0.00014, "output": 0.00028},
    },
    "together": {
        "llama-3.1-70b": {"input": 0.0009, "output": 0.0009},
        "llama-3.1-8b": {"input": 0.0002, "output": 0.0002},
        "mixtral-8x7b": {"input": 0.0006, "output": 0.0006},
        "default": {"input": 0.0005, "output": 0.0005},
    },
    "qwen": {
        "qwen-turbo": {"input": 0.0008, "output": 0.002},
        "qwen-plus": {"input": 0.004, "output": 0.012},
        "qwen-max": {"input": 0.02, "output": 0.06},
        "default": {"input": 0.001, "output": 0.002},
    },
    "glm": {
        "glm-4": {"input": 0.014, "output": 0.014},
        "glm-4-flash": {"input": 0.0001, "output": 0.0001},
        "glm-3-turbo": {"input": 0.0007, "output": 0.0007},
        "default": {"input": 0.001, "output": 0.001},
    },
    "kimi": {
        "moonshot-v1-8k": {"input": 0.0017, "output": 0.0017},
        "moonshot-v1-32k": {"input": 0.0034, "output": 0.0034},
        "moonshot-v1-128k": {"input": 0.0085, "output": 0.0085},
        "default": {"input": 0.002, "output": 0.002},
    },
    "megallm": {
        "default": {"input": 0.001, "output": 0.002},
    },
    # Local providers have no cost
    "ollama": {
        "default": {"input": 0.0, "output": 0.0},
    },
}


@dataclass
class UsageRecord:
    """Record of AI provider usage for cost tracking."""
    
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost: float | None
    latency_ms: float
    success: bool
    raw_input: str | None = None
    raw_output: str | None = None
    user_id: int | None = None
    extra: dict[str, Any] | None = None


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """
    Estimate the cost of an AI request based on provider and token usage.
    
    Args:
        provider: The AI provider name
        model: The model name used
        input_tokens: Number of input tokens (None if unknown)
        output_tokens: Number of output tokens (None if unknown)
        
    Returns:
        Estimated cost in USD, or None if tokens are unknown
    """
    if input_tokens is None or output_tokens is None:
        return None
    
    provider_lower = provider.lower()
    model_lower = model.lower()
    
    # Get provider costs
    provider_costs = PROVIDER_COSTS.get(provider_lower, {})
    
    # Try to find exact model match, otherwise use default
    model_costs = None
    for model_key, costs in provider_costs.items():
        if model_key != "default" and model_key in model_lower:
            model_costs = costs
            break
    
    if model_costs is None:
        model_costs = provider_costs.get("default", {"input": 0.0, "output": 0.0})
    
    # Calculate cost (costs are per 1K tokens)
    input_cost = (input_tokens / 1000) * model_costs.get("input", 0.0)
    output_cost = (output_tokens / 1000) * model_costs.get("output", 0.0)
    
    return round(input_cost + output_cost, 6)


def create_usage_record_from_response(
    response: AIResponse,
    raw_input: str | None = None,
    user_id: int | None = None,
    success: bool = True,
) -> UsageRecord:
    """
    Create a UsageRecord from an AIResponse.
    
    Args:
        response: The AI response object
        raw_input: The original input text (optional)
        user_id: The user ID making the request (optional)
        success: Whether the request was successful
        
    Returns:
        UsageRecord ready for persistence
    """
    input_tokens = response.usage.get("input_tokens") if response.usage else None
    output_tokens = response.usage.get("output_tokens") if response.usage else None
    
    estimated_cost = estimate_cost(
        provider=response.provider,
        model=response.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    
    return UsageRecord(
        provider=response.provider,
        model=response.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
        latency_ms=response.latency_ms,
        success=success,
        raw_input=raw_input,
        raw_output=response.content[:1000] if response.content else None,  # Truncate for storage
        user_id=user_id,
    )


class CostTracker:
    """
    Service for tracking AI provider usage and costs.
    
    Records usage data to the backend-api's ai_audit table for
    monitoring, cost analysis, and usage reporting.
    """
    
    def __init__(
        self,
        backend_api_url: str,
        enabled: bool = True,
        timeout: float = 5.0,
    ):
        """
        Initialize the cost tracker.
        
        Args:
            backend_api_url: URL of the backend API service
            enabled: Whether cost tracking is enabled
            timeout: HTTP request timeout in seconds
        """
        self.backend_api_url = backend_api_url.rstrip("/")
        self.enabled = enabled
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def record_usage(self, record: UsageRecord) -> bool:
        """
        Record AI usage to the backend API.
        
        Args:
            record: The usage record to persist
            
        Returns:
            True if recording succeeded, False otherwise
        """
        if not self.enabled:
            logger.debug("Cost tracking disabled, skipping record")
            return True
        
        try:
            client = await self._get_client()
            
            payload = {
                "provider": record.provider,
                "model_name": record.model,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "estimated_cost": record.estimated_cost,
                "latency_ms": record.latency_ms,
                "success": record.success,
                "raw_input": record.raw_input,
                "raw_output": record.raw_output,
                "user_id": record.user_id,
                "extra": record.extra,
            }
            
            response = await client.post(
                f"{self.backend_api_url}/api/v1/ai-audit",
                json=payload,
            )
            
            if response.status_code in (200, 201):
                logger.debug(
                    "Usage recorded successfully",
                    provider=record.provider,
                    model=record.model,
                    cost=record.estimated_cost,
                )
                return True
            else:
                logger.warning(
                    "Failed to record usage",
                    status_code=response.status_code,
                    response=response.text[:200],
                )
                return False
                
        except httpx.TimeoutException:
            logger.warning("Timeout recording usage to backend API")
            return False
        except httpx.RequestError as e:
            logger.warning(
                "Error recording usage to backend API",
                error=str(e),
            )
            return False
        except Exception as e:
            logger.error(
                "Unexpected error recording usage",
                error=str(e),
            )
            return False
    
    async def record_from_response(
        self,
        response: AIResponse,
        raw_input: str | None = None,
        user_id: int | None = None,
        success: bool = True,
    ) -> bool:
        """
        Convenience method to record usage directly from an AIResponse.
        
        Args:
            response: The AI response object
            raw_input: The original input text (optional)
            user_id: The user ID making the request (optional)
            success: Whether the request was successful
            
        Returns:
            True if recording succeeded, False otherwise
        """
        record = create_usage_record_from_response(
            response=response,
            raw_input=raw_input,
            user_id=user_id,
            success=success,
        )
        return await self.record_usage(record)


# Singleton instance
_cost_tracker: CostTracker | None = None


def get_cost_tracker(
    backend_api_url: str | None = None,
    enabled: bool = True,
) -> CostTracker:
    """
    Get or create the cost tracker singleton.
    
    Args:
        backend_api_url: URL of the backend API (required on first call)
        enabled: Whether cost tracking is enabled
        
    Returns:
        The CostTracker instance
    """
    global _cost_tracker
    
    if _cost_tracker is None:
        if backend_api_url is None:
            raise ValueError("backend_api_url is required on first call")
        _cost_tracker = CostTracker(
            backend_api_url=backend_api_url,
            enabled=enabled,
        )
    
    return _cost_tracker
