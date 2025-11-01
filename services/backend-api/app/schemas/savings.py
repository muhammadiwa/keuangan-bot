from typing import Literal

from pydantic import BaseModel, Field


class SavingsAccountCreate(BaseModel):
    user_id: int
    name: str
    target_amount: float = Field(..., gt=0)
    currency: str = "IDR"


class SavingsAccountResponse(BaseModel):
    id: int
    name: str
    target_amount: float
    current_amount: float
    currency: str = "IDR"


class SavingsTransactionRequest(BaseModel):
    user_id: int
    saving_name: str
    amount: float = Field(..., gt=0)
    direction: Literal["deposit", "withdraw"]
    note: str | None = None


class SavingsTransactionResponse(BaseModel):
    message: str
