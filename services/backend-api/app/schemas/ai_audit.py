"""
Schemas for AI audit/cost tracking.

Requirements: 14.1, 14.2
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AIAuditCreate(BaseModel):
    """Request schema for creating an AI audit record."""
    
    provider: str | None = Field(None, description="AI provider name (e.g., openai, anthropic)")
    model_name: str | None = Field(None, description="Model name used")
    input_tokens: int | None = Field(None, ge=0, description="Number of input tokens")
    output_tokens: int | None = Field(None, ge=0, description="Number of output tokens")
    estimated_cost: float | None = Field(None, ge=0, description="Estimated cost in USD")
    latency_ms: float | None = Field(None, ge=0, description="Response latency in milliseconds")
    success: bool = Field(True, description="Whether the request was successful")
    raw_input: str | None = Field(None, description="Raw input text (truncated)")
    raw_output: str | None = Field(None, description="Raw output text (truncated)")
    user_id: int | None = Field(None, description="User ID if available")
    extra: dict[str, Any] | None = Field(None, description="Additional metadata")


class AIAuditResponse(BaseModel):
    """Response schema for AI audit record."""
    
    id: int
    provider: str | None
    model_name: str | None
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost: float | None
    latency_ms: float | None
    success: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class AIUsageSummary(BaseModel):
    """Summary of AI usage for reporting."""
    
    provider: str
    total_requests: int
    successful_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost: float
    avg_latency_ms: float
