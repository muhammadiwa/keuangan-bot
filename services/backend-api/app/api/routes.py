from datetime import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.db.session import get_session
from app.schemas.ai_audit import AIAuditCreate, AIAuditResponse
from app.schemas.reports import CategoryBreakdown, DailyReportResponse
from app.schemas.savings import (
    SavingsAccountCreate,
    SavingsAccountResponse,
    SavingsTransactionRequest,
    SavingsTransactionResponse,
)
from app.schemas.transactions import Transaction, TransactionCreate, TransactionList
from app.schemas.wa import IncomingMessage, WAResponse
from app.services.wa import handle_incoming_message

router = APIRouter()


@router.get("/healthz", response_model=dict)
async def healthz() -> dict:
    return {"status": "ok"}


@router.post("/wa/incoming", response_model=WAResponse)
async def wa_incoming(
    payload: IncomingMessage, session: AsyncSession = Depends(get_session)
) -> WAResponse:
    return await handle_incoming_message(session=session, payload=payload)


@router.post("/transactions", response_model=Transaction, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    body: TransactionCreate, session: AsyncSession = Depends(get_session)
) -> Transaction:
    tx = models.Transaction(
        user_id=body.user_id,
        direction=body.direction,
        amount=Decimal(str(body.amount)),
        currency=body.currency,
        category_id=body.category_id,
        description=body.description,
        tx_datetime=body.tx_datetime,
        source=body.source,
        raw_text=body.raw_text,
        ai_confidence=body.ai_confidence,
        status=body.status,
    )
    session.add(tx)
    await session.commit()
    await session.refresh(tx)
    return _transaction_to_schema(tx)


@router.get("/transactions", response_model=TransactionList)
async def list_transactions(
    user_id: int,
    start: Annotated[datetime | None, Query()] = None,
    end: Annotated[datetime | None, Query()] = None,
    category_id: Annotated[int | None, Query()] = None,
    session: AsyncSession = Depends(get_session),
) -> TransactionList:
    filters = [models.Transaction.user_id == user_id]
    if start is not None:
        filters.append(models.Transaction.tx_datetime >= start)
    if end is not None:
        filters.append(models.Transaction.tx_datetime <= end)
    if category_id is not None:
        filters.append(models.Transaction.category_id == category_id)

    stmt = (
        select(models.Transaction)
        .where(and_(*filters))
        .order_by(models.Transaction.tx_datetime.desc())
    )
    result = await session.execute(stmt)
    items = [_transaction_to_schema(tx) for tx in result.scalars().all()]
    total = len(items)
    return TransactionList(items=items, total=total)


@router.post("/savings/accounts", response_model=SavingsAccountResponse, status_code=status.HTTP_201_CREATED)
async def create_savings_account(
    body: SavingsAccountCreate, session: AsyncSession = Depends(get_session)
) -> SavingsAccountResponse:
    existing_stmt = select(models.SavingsAccount).where(
        models.SavingsAccount.user_id == body.user_id,
        func.lower(models.SavingsAccount.name) == body.name.lower(),
    )
    existing = await session.scalar(existing_stmt)
    if existing:
        raise HTTPException(status_code=400, detail="Tabungan dengan nama tersebut sudah ada")

    account = models.SavingsAccount(
        user_id=body.user_id,
        name=body.name,
        target_amount=Decimal(str(body.target_amount)),
        current_amount=Decimal("0"),
        currency=body.currency,
    )
    session.add(account)
    await session.commit()
    await session.refresh(account)
    return SavingsAccountResponse(
        id=account.id,
        name=account.name,
        target_amount=float(account.target_amount),
        current_amount=float(account.current_amount),
        currency=account.currency,
    )


@router.post("/savings/deposit", response_model=SavingsTransactionResponse)
async def deposit_savings(
    body: SavingsTransactionRequest, session: AsyncSession = Depends(get_session)
) -> SavingsTransactionResponse:
    if body.direction != "deposit":
        raise HTTPException(status_code=400, detail="direction must be deposit")
    return await _apply_savings_movement(body, session=session)


@router.post("/savings/withdraw", response_model=SavingsTransactionResponse)
async def withdraw_savings(
    body: SavingsTransactionRequest, session: AsyncSession = Depends(get_session)
) -> SavingsTransactionResponse:
    if body.direction != "withdraw":
        raise HTTPException(status_code=400, detail="direction must be withdraw")
    return await _apply_savings_movement(body, session=session)


@router.get("/reports/daily", response_model=DailyReportResponse)
async def get_daily_report(
    user_id: int,
    report_date: datetime | None = None,
    session: AsyncSession = Depends(get_session),
) -> DailyReportResponse:
    report_date = report_date or datetime.utcnow()
    start = datetime.combine(report_date.date(), datetime.min.time(), tzinfo=report_date.tzinfo)
    end = datetime.combine(report_date.date(), datetime.max.time(), tzinfo=report_date.tzinfo)

    tx_stmt = select(models.Transaction).where(
        models.Transaction.user_id == user_id,
        models.Transaction.tx_datetime >= start,
        models.Transaction.tx_datetime <= end,
    )
    tx_result = await session.execute(tx_stmt)
    transactions = [row[0] for row in tx_result.all()]

    total_income = sum(float(t.amount) for t in transactions if t.direction == "income")
    total_expense = sum(float(t.amount) for t in transactions if t.direction == "expense")

    breakdown_stmt = (
        select(models.Category.name, func.sum(models.Transaction.amount))
        .join(models.Category, models.Transaction.category_id == models.Category.id, isouter=True)
        .where(
            models.Transaction.user_id == user_id,
            models.Transaction.direction == "expense",
            models.Transaction.tx_datetime >= start,
            models.Transaction.tx_datetime <= end,
        )
        .group_by(models.Category.name)
    )
    breakdown_result = await session.execute(breakdown_stmt)
    category_breakdown = [
        CategoryBreakdown(category=name or "Lainnya", amount=float(amount))
        for name, amount in breakdown_result.all()
    ]

    month_start = datetime(report_date.year, report_date.month, 1, tzinfo=report_date.tzinfo)
    next_month = datetime(
        report_date.year + (report_date.month // 12),
        (report_date.month % 12) + 1,
        1,
        tzinfo=report_date.tzinfo,
    )
    month_stmt = select(models.Transaction).where(
        models.Transaction.user_id == user_id,
        models.Transaction.tx_datetime >= month_start,
        models.Transaction.tx_datetime < next_month,
    )
    month_result = await session.execute(month_stmt)
    month_transactions = [row[0] for row in month_result.all()]
    month_income = sum(float(t.amount) for t in month_transactions if t.direction == "income")
    month_expense = sum(float(t.amount) for t in month_transactions if t.direction == "expense")

    return DailyReportResponse(
        date=report_date.date(),
        total_income=total_income,
        total_expense=total_expense,
        category_breakdown=category_breakdown,
        month_balance=month_income - month_expense,
    )


def _transaction_to_schema(tx: models.Transaction) -> Transaction:
    return Transaction(
        id=tx.id,
        user_id=tx.user_id,
        direction=tx.direction,
        amount=float(tx.amount),
        currency=tx.currency,
        category_id=tx.category_id,
        description=tx.description,
        tx_datetime=tx.tx_datetime,
        source=tx.source,
        raw_text=tx.raw_text,
        ai_confidence=tx.ai_confidence,
        status=tx.status,
    )


async def _apply_savings_movement(
    body: SavingsTransactionRequest, session: AsyncSession
) -> SavingsTransactionResponse:
    account_stmt = select(models.SavingsAccount).where(
        models.SavingsAccount.user_id == body.user_id,
        func.lower(models.SavingsAccount.name) == body.saving_name.lower(),
    )
    account = await session.scalar(account_stmt)
    if not account:
        raise HTTPException(status_code=404, detail="Tabungan tidak ditemukan")

    amount_decimal = Decimal(str(body.amount))
    if body.direction == "withdraw" and account.current_amount - amount_decimal < Decimal("0"):
        raise HTTPException(status_code=400, detail="Saldo tabungan tidak mencukupi")

    movement = models.SavingsTransaction(
        savings_account_id=account.id,
        user_id=account.user_id,
        direction=body.direction,
        amount=amount_decimal,
        note=body.note,
        source="whatsapp",
    )
    session.add(movement)

    if body.direction == "deposit":
        account.current_amount += amount_decimal
    else:
        account.current_amount -= amount_decimal

    await session.commit()
    message = (
        f"✅ Setoran Rp{body.amount:,.0f} ke tabungan {account.name}."
        if body.direction == "deposit"
        else f"✅ Penarikan Rp{body.amount:,.0f} dari tabungan {account.name}."
    )
    return SavingsTransactionResponse(message=message)


# AI Audit endpoints for cost tracking (Requirements: 14.1, 14.2)

@router.post("/api/v1/ai-audit", response_model=AIAuditResponse, status_code=status.HTTP_201_CREATED)
async def create_ai_audit(
    body: AIAuditCreate, session: AsyncSession = Depends(get_session)
) -> AIAuditResponse:
    """
    Record AI provider usage for cost tracking.
    
    This endpoint is called by ai-media-service after each AI provider call
    to record token usage, costs, and latency metrics.
    """
    audit = models.AIAudit(
        user_id=body.user_id,
        provider=body.provider,
        model_name=body.model_name,
        raw_input=body.raw_input,
        raw_output=body.raw_output,
        success=body.success,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        estimated_cost=Decimal(str(body.estimated_cost)) if body.estimated_cost is not None else None,
        latency_ms=body.latency_ms,
        extra=body.extra,
    )
    session.add(audit)
    await session.commit()
    await session.refresh(audit)
    
    return AIAuditResponse(
        id=audit.id,
        provider=audit.provider,
        model_name=audit.model_name,
        input_tokens=audit.input_tokens,
        output_tokens=audit.output_tokens,
        estimated_cost=float(audit.estimated_cost) if audit.estimated_cost else None,
        latency_ms=audit.latency_ms,
        success=audit.success,
        created_at=audit.created_at,
    )
