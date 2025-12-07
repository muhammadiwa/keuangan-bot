"""Add cost tracking columns to ai_audit table

Revision ID: 20241207_01
Revises: 20240519_01
Create Date: 2024-12-07

This migration adds columns for tracking AI provider usage and costs:
- provider: The AI provider used (ollama, openai, anthropic, etc.)
- input_tokens: Number of input tokens consumed
- output_tokens: Number of output tokens generated
- estimated_cost: Estimated cost in USD
- latency_ms: Response latency in milliseconds
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20241207_01"
down_revision = "20240519_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to ai_audit table for cost tracking
    op.add_column(
        "ai_audit",
        sa.Column("provider", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "ai_audit",
        sa.Column("input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "ai_audit",
        sa.Column("output_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "ai_audit",
        sa.Column("estimated_cost", sa.Numeric(12, 6), nullable=True),
    )
    op.add_column(
        "ai_audit",
        sa.Column("latency_ms", sa.Float(), nullable=True),
    )
    
    # Add index on provider for aggregation queries
    op.create_index("ix_ai_audit_provider", "ai_audit", ["provider"])


def downgrade() -> None:
    op.drop_index("ix_ai_audit_provider", table_name="ai_audit")
    op.drop_column("ai_audit", "latency_ms")
    op.drop_column("ai_audit", "estimated_cost")
    op.drop_column("ai_audit", "output_tokens")
    op.drop_column("ai_audit", "input_tokens")
    op.drop_column("ai_audit", "provider")
