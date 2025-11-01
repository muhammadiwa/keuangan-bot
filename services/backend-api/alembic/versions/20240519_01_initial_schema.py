"""Initial schema for finance bot"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20240519_01"
down_revision = None
branch_labels = None
depends_on = None


transaction_direction = sa.Enum("income", "expense", "transfer", name="transaction_direction")
transaction_source = sa.Enum("whatsapp", "web", "import", name="transaction_source")
transaction_status = sa.Enum("confirmed", "pending", name="transaction_status")
wa_direction = sa.Enum("in", "out", name="wa_direction")
savings_direction = sa.Enum("deposit", "withdraw", name="savings_direction")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("phone", sa.String(length=32), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(length=255), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_users_phone", "users", ["phone"])

    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("keywords", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("user_id", "name", name="uq_categories_user_name"),
    )
    op.create_index("ix_categories_user_id", "categories", ["user_id"])

    op.create_table(
        "savings_accounts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("target_amount", sa.Numeric(18, 2), nullable=False),
        sa.Column("current_amount", sa.Numeric(18, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default=sa.text("'IDR'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("user_id", "name", name="uq_savings_user_name"),
    )
    op.create_index("ix_savings_accounts_user_id", "savings_accounts", ["user_id"])

    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("direction", transaction_direction, nullable=False),
        sa.Column("amount", sa.Numeric(18, 2), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default=sa.text("'IDR'")),
        sa.Column("category_id", sa.Integer(), sa.ForeignKey("categories.id", ondelete="SET NULL"), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("tx_datetime", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", transaction_source, nullable=False, server_default=sa.text("'whatsapp'")),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("ai_confidence", sa.Float(), nullable=True),
        sa.Column("status", transaction_status, nullable=False, server_default=sa.text("'confirmed'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_transactions_user_id", "transactions", ["user_id"])
    op.create_index("ix_transactions_category_id", "transactions", ["category_id"])
    op.create_index("ix_transactions_tx_datetime", "transactions", ["tx_datetime"])
    op.create_index("ix_transactions_user_txdt", "transactions", ["user_id", "tx_datetime"])

    op.create_table(
        "savings_transactions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("savings_account_id", sa.Integer(), sa.ForeignKey("savings_accounts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("direction", savings_direction, nullable=False),
        sa.Column("amount", sa.Numeric(18, 2), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("amount > 0", name="ck_savings_transactions_amount_positive"),
    )
    op.create_index("ix_savings_transactions_account_id", "savings_transactions", ["savings_account_id"])
    op.create_index("ix_savings_transactions_user_id", "savings_transactions", ["user_id"])

    op.create_table(
        "wa_messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("wa_from", sa.String(length=32), nullable=False),
        sa.Column("wa_body", sa.Text(), nullable=True),
        sa.Column("direction", wa_direction, nullable=False),
        sa.Column("intent", sa.String(length=64), nullable=True),
        sa.Column("response", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_wa_messages_user_id", "wa_messages", ["user_id"])
    op.create_index("ix_wa_messages_wa_from", "wa_messages", ["wa_from"])
    op.create_index("ix_wa_messages_created_at", "wa_messages", ["created_at"])

    op.create_table(
        "ai_audit",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("model_name", sa.String(length=120), nullable=True),
        sa.Column("raw_input", sa.Text(), nullable=True),
        sa.Column("raw_output", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("extra", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index("ix_ai_audit_user_id", "ai_audit", ["user_id"])
    op.create_index("ix_ai_audit_created_at", "ai_audit", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_ai_audit_created_at", table_name="ai_audit")
    op.drop_index("ix_ai_audit_user_id", table_name="ai_audit")
    op.drop_table("ai_audit")

    op.drop_index("ix_wa_messages_created_at", table_name="wa_messages")
    op.drop_index("ix_wa_messages_wa_from", table_name="wa_messages")
    op.drop_index("ix_wa_messages_user_id", table_name="wa_messages")
    op.drop_table("wa_messages")
    wa_direction.drop(op.get_bind(), checkfirst=False)

    op.drop_index("ix_savings_transactions_user_id", table_name="savings_transactions")
    op.drop_index("ix_savings_transactions_account_id", table_name="savings_transactions")
    op.drop_table("savings_transactions")
    savings_direction.drop(op.get_bind(), checkfirst=False)

    op.drop_index("ix_transactions_user_txdt", table_name="transactions")
    op.drop_index("ix_transactions_tx_datetime", table_name="transactions")
    op.drop_index("ix_transactions_category_id", table_name="transactions")
    op.drop_index("ix_transactions_user_id", table_name="transactions")
    op.drop_table("transactions")
    transaction_status.drop(op.get_bind(), checkfirst=False)
    transaction_source.drop(op.get_bind(), checkfirst=False)
    transaction_direction.drop(op.get_bind(), checkfirst=False)

    op.drop_index("ix_savings_accounts_user_id", table_name="savings_accounts")
    op.drop_table("savings_accounts")

    op.drop_index("ix_categories_user_id", table_name="categories")
    op.drop_table("categories")

    op.drop_index("ix_users_phone", table_name="users")
    op.drop_table("users")
