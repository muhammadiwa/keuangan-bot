from datetime import date

from pydantic import BaseModel


class DailyReportQuery(BaseModel):
    user_id: int
    date: date


class CategoryBreakdown(BaseModel):
    category: str
    amount: float


class DailyReportResponse(BaseModel):
    date: date
    total_income: float
    total_expense: float
    category_breakdown: list[CategoryBreakdown]
    month_balance: float
