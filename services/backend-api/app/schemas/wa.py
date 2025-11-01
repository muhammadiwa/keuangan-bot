from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class IncomingMessage(BaseModel):
    from_number: str = Field(..., example="6281234567890")
    message_type: Literal["text", "audio", "image"]
    text: Optional[str] = None
    media_url: Optional[str] = None
    timestamp: datetime


class WAResponse(BaseModel):
    status: str = "ok"
    reply: str
