from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TransactionBase(BaseModel):
    user_id: int
    direction: Literal["income", "expense", "transfer"]
    amount: float = Field(..., gt=0)
    currency: str = "IDR"
    category_id: Optional[int] = None
    description: Optional[str] = None
    tx_datetime: datetime = Field(default_factory=datetime.utcnow)
    source: Literal["whatsapp", "web", "import"] = "whatsapp"
    raw_text: Optional[str] = None
    ai_confidence: Optional[float] = Field(None, ge=0, le=1)
    status: Literal["confirmed", "pending"] = "confirmed"


class TransactionCreate(TransactionBase):
    pass


class Transaction(TransactionBase):
    id: int

    class Config:
        from_attributes = True


class TransactionList(BaseModel):
    items: list[Transaction]
    total: int
