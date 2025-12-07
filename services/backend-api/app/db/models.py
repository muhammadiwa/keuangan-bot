from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)

try:  # pragma: no cover - fallback for non-Postgres engines during local dev
    from sqlalchemy.dialects.postgresql import JSONB as JSONType
except ModuleNotFoundError:  # pragma: no cover
    from sqlalchemy import JSON as JSONType
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    timezone: Mapped[str | None] = mapped_column(String(64))

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="user")
    savings_accounts: Mapped[list["SavingsAccount"]] = relationship(back_populates="user")


class Category(Base, TimestampMixin):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    keywords: Mapped[str | None] = mapped_column(Text)

    user: Mapped[User | None] = relationship(backref="categories")

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_categories_user_name"),
        Index("ix_categories_user_id", "user_id"),
    )


class Transaction(Base, TimestampMixin):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    direction: Mapped[str] = mapped_column(
        Enum("income", "expense", "transfer", name="transaction_direction"),
        nullable=False,
    )
    amount: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), default="IDR")
    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("categories.id", ondelete="SET NULL"), index=True
    )
    description: Mapped[str | None] = mapped_column(Text)
    tx_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(
        Enum("whatsapp", "web", "import", name="transaction_source"), default="whatsapp"
    )
    raw_text: Mapped[str | None] = mapped_column(Text)
    ai_confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(
        Enum("confirmed", "pending", name="transaction_status"), default="confirmed"
    )

    user: Mapped[User] = relationship(back_populates="transactions")
    category: Mapped[Category | None] = relationship()

    __table_args__ = (
        Index("ix_transactions_user_txdt", "user_id", "tx_datetime"),
    )


class SavingsAccount(Base, TimestampMixin):
    __tablename__ = "savings_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    target_amount: Mapped[float] = mapped_column(Numeric(18, 2))
    current_amount: Mapped[float] = mapped_column(Numeric(18, 2), default=0)
    currency: Mapped[str] = mapped_column(String(8), default="IDR")

    user: Mapped[User] = relationship(back_populates="savings_accounts")
    transactions: Mapped[list["SavingsTransaction"]] = relationship(
        back_populates="savings_account", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_savings_user_name"),
    )


class SavingsTransaction(Base):
    __tablename__ = "savings_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    savings_account_id: Mapped[int] = mapped_column(
        ForeignKey("savings_accounts.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    direction: Mapped[str] = mapped_column(
        Enum("deposit", "withdraw", name="savings_direction"), nullable=False
    )
    amount: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    savings_account: Mapped[SavingsAccount] = relationship(back_populates="transactions")
    user: Mapped[User] = relationship()

    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_savings_transactions_amount_positive"),
    )


class WAMessage(Base):
    __tablename__ = "wa_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    wa_from: Mapped[str] = mapped_column(String(32), index=True)
    wa_body: Mapped[str | None] = mapped_column(Text)
    direction: Mapped[str] = mapped_column(Enum("in", "out", name="wa_direction"), nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64))
    response: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True
    )

    user: Mapped[User | None] = relationship()


class AIAudit(Base):
    __tablename__ = "ai_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    provider: Mapped[str | None] = mapped_column(String(64))
    model_name: Mapped[str | None] = mapped_column(String(120))
    raw_input: Mapped[str | None] = mapped_column(Text)
    raw_output: Mapped[str | None] = mapped_column(Text)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost: Mapped[float | None] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True
    )
    extra: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    user: Mapped[User | None] = relationship()
