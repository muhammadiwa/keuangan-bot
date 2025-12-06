from __future__ import annotations

import calendar
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from loguru import logger
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import models
from app.schemas.wa import IncomingMessage, WAResponse

settings = get_settings()


@dataclass
class PendingAction:
    action_type: str
    data: dict[str, Any]
    expires_at: datetime


@dataclass
class PendingActionResult:
    reply: str
    intent: str


_pending_actions: dict[int, PendingAction] = {}


class ParsedIntent:
    def __init__(
        self,
        intent: str,
        amount: float | None = None,
        currency: str | None = "IDR",
        description: str | None = None,
        category_suggestion: str | None = None,
        timestamp: datetime | None = None,
        direction: str | None = None,
        confidence: float | None = None,
    ) -> None:
        self.intent = intent
        self.amount = amount
        self.currency = currency
        self.description = description
        self.category_suggestion = category_suggestion
        self.timestamp = timestamp
        self.direction = direction
        self.confidence = confidence

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ParsedIntent":
        return cls(
            intent=str(data.get("intent", "unknown")),
            amount=float(data["amount"]) if data.get("amount") is not None else None,
            currency=data.get("currency") or "IDR",
            description=data.get("description") or None,
            category_suggestion=data.get("category_suggestion") or None,
            timestamp=_parse_datetime(data.get("datetime")),
            direction=data.get("direction") or None,
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        )


async def handle_incoming_message(session: AsyncSession, payload: IncomingMessage) -> WAResponse:
    logger.info("Processing incoming WA message", payload=payload.model_dump())
    user = await _get_or_create_user(session, payload.from_number)

    await _log_wa_message(
        session,
        user_id=user.id,
        wa_from=payload.from_number,
        body=payload.text or payload.media_url or "",
        direction="in",
    )

    text_content: str | None = None
    pending_result = await _maybe_handle_pending_action(session, user.id, payload)
    if pending_result is not None:
        await _log_wa_message(
            session,
            user_id=user.id,
            wa_from=payload.from_number,
            body=pending_result.reply,
            direction="out",
            intent=pending_result.intent,
            response=None,
        )
        await session.commit()
        return WAResponse(reply=pending_result.reply)

    try:
        text_content = await _extract_text_from_payload(payload)
        logger.debug("Extracted text", text=text_content)

        parsed, audit_payload, model_name = await _parse_intent_with_ai(text_content)

        if _text_requests_savings_list(text_content):
            parsed.intent = "list_savings"
            parsed.description = None
            parsed.category_suggestion = None
            parsed.amount = None
            parsed.direction = None
        if _text_requests_balance(text_content):
            parsed.intent = "get_balance"
            parsed.description = None
            parsed.category_suggestion = None
            parsed.amount = None
            parsed.direction = None
        if _text_requests_category_list(text_content):
            parsed.intent = "list_categories"
            parsed.description = None
            parsed.category_suggestion = None
            parsed.amount = None
            parsed.direction = None
        if _text_requests_category_add(text_content):
            parsed.intent = "add_category"
        if _text_requests_category_delete(text_content):
            parsed.intent = "delete_category"
        if _text_requests_category_rename(text_content):
            parsed.intent = "rename_category"
        if _text_requests_transactions_list(text_content):
            parsed.intent = "list_transactions"
            parsed.description = None
            parsed.category_suggestion = None
            parsed.amount = None
            parsed.direction = None
        if (
            parsed.intent == "create_transaction"
            and parsed.amount is None
            and _text_mentions_saving(text_content)
        ):
            parsed.intent = "create_saving"
            parsed.description = parsed.description or text_content
            parsed.category_suggestion = None
            parsed.direction = None
        if parsed.intent in {"create_saving", "deposit_saving", "withdraw_saving"}:
            reference = _extract_saving_reference(text_content)
            if reference:
                parsed.description = reference
        if parsed.intent in {"add_category", "delete_category", "rename_category"}:
            map_action = {
                "add_category": "add",
                "delete_category": "delete",
                "rename_category": "rename",
            }
            first, second = _extract_category_names(text_content, map_action[parsed.intent])
            if parsed.intent in {"add_category", "delete_category"} and first:
                parsed.description = first
            elif parsed.intent == "rename_category" and first and second:
                parsed.description = f"{first}|{second}"

        reply: str
        intent = parsed.intent
        if intent == "create_transaction":
            reply = await _handle_transaction_intent(
                session, user.id, payload, parsed, text_content
            )
        elif intent == "create_saving":
            reply = await _handle_create_saving(session, user.id, parsed)
        elif intent in {"deposit_saving", "withdraw_saving"}:
            reply = await _handle_savings_movement(session, user.id, parsed)
        elif intent == "list_savings":
            reply = await _handle_list_savings(session, user.id)
        elif intent == "get_balance":
            reply = await _handle_balance_request(session, user.id)
        elif intent == "list_categories":
            reply = await _handle_list_categories(session, user.id)
        elif intent == "add_category":
            reply = await _handle_add_category(session, user.id, text_content)
        elif intent == "rename_category":
            reply = await _handle_rename_category(session, user.id, text_content)
        elif intent == "delete_category":
            reply = await _handle_delete_category(session, user.id, text_content)
        elif intent == "list_transactions":
            reply = await _handle_transactions_request(session, user.id, payload)
        elif intent == "get_report":
            reply = await _handle_report_request(session, user.id, payload)
        else:
            reply = "Maaf, aku belum paham. Bisa jelaskan lagi?"

        await _log_ai_audit(session, user.id, text_content, audit_payload, model_name, success=True)
        await _log_wa_message(
            session,
            user_id=user.id,
            wa_from=payload.from_number,
            body=reply,
            direction="out",
            intent=intent,
            response=json.dumps(audit_payload),
        )

        await session.commit()
        return WAResponse(reply=reply)
    except Exception as exc:
        await session.rollback()
        logger.exception("Failed processing incoming message", error=str(exc))
        error_reply = "Terjadi kesalahan saat memproses pesan. Coba lagi ya."
        await _log_ai_audit(
            session,
            user.id,
            text_content or (payload.text or ""),
            {"error": str(exc)},
            "ai-media-service:nlu",
            success=False,
        )
        await _log_wa_message(
            session,
            user_id=user.id,
            wa_from=payload.from_number,
            body=error_reply,
            direction="out",
            intent="error",
            response=str(exc),
        )
        await session.commit()
        return WAResponse(reply=error_reply)


async def _get_or_create_user(session: AsyncSession, phone: str) -> models.User:
    stmt = select(models.User).where(models.User.phone == phone)
    user = await session.scalar(stmt)
    if user:
        return user

    user = models.User(phone=phone, name=phone, timezone=settings.timezone)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def _extract_text_from_payload(payload: IncomingMessage) -> str:
    if payload.message_type == "text":
        return payload.text or ""

    if not payload.media_url:
        raise ValueError("media_url required for non-text message")

    if payload.message_type == "audio":
        stt_response = await _call_ai_service("/media/stt", {"media_url": payload.media_url})
        return stt_response.get("text", "")

    if payload.message_type == "image":
        ocr_response = await _call_ai_service("/media/ocr", {"media_url": payload.media_url})
        formatted = _format_ocr_response(ocr_response)
        return formatted

    return payload.text or ""


async def _parse_intent_with_ai(text: str) -> tuple[ParsedIntent, dict[str, object], str]:
    try:
        response = await _call_ai_service("/ai/parse", {"text": text})
        parsed = ParsedIntent.from_dict(response)
        model_name = response.get("model") if isinstance(response, dict) else None
        return parsed, response, model_name or "ai-media-service:nlu"
    except Exception as exc:  # pragma: no cover - network failures fallback
        logger.warning("AI parse failed, using heuristic fallback", error=str(exc))
        fallback = _heuristic_parse(text)
        return fallback, {
            "intent": fallback.intent,
            "amount": fallback.amount,
            "currency": fallback.currency,
            "description": fallback.description,
            "category_suggestion": fallback.category_suggestion,
            "datetime": fallback.timestamp.isoformat() if fallback.timestamp else None,
            "fallback": True,
        }, "heuristic"


async def _handle_transaction_intent(
    session: AsyncSession,
    user_id: int,
    payload: IncomingMessage,
    parsed: ParsedIntent,
    raw_text: str,
) -> str:
    if parsed.amount is None and not _message_looks_like_transaction(raw_text):
        return (
            "Sepertinya itu bukan catatan transaksi. Coba jelaskan seperti \"beli kopi 20rb\" atau \"terima gaji 5jt\" ya."
        )

    if parsed.amount is None:
        return "Hmm, aku belum menangkap nominalnya. Bisa sebutkan jumlahnya supaya aku bantu catat ya?"

    direction = parsed.direction or _detect_direction(raw_text)

    if _looks_like_amount_only(raw_text):
        amount_value = float(parsed.amount)
        currency_value = parsed.currency or "IDR"
        _set_pending_action(
            user_id=user_id,
            action_type="create_transaction_details",
            data={
                "amount": amount_value,
                "currency": currency_value,
                "raw_text": raw_text,
            },
        )
        return (
            f"Nominal {_format_currency(amount_value, currency_value)} sudah dicatat. "
            "Boleh jelaskan ini untuk apa dan apakah masuk atau keluar? "
            "Contoh: \"beli pulsa 50rb\" atau \"terima gaji 50rb\"."
        )

    category_id = await _resolve_category(session, user_id, parsed.category_suggestion)

    transaction = models.Transaction(
        user_id=user_id,
        direction=direction,
        amount=Decimal(str(parsed.amount)),
        currency=parsed.currency or "IDR",
        category_id=category_id,
        description=parsed.description or raw_text,
        tx_datetime=parsed.timestamp or payload.timestamp,
        source="whatsapp",
        raw_text=raw_text,
        ai_confidence=parsed.confidence,
        status="confirmed",
    )
    session.add(transaction)
    await session.flush()

    category_name = await _get_category_name(session, category_id)
    summary = await _build_transaction_summary(
        session=session,
        user_id=user_id,
        transaction=transaction,
        category_name=category_name,
    )
    return summary


async def _handle_create_saving(
    session: AsyncSession, user_id: int, parsed: ParsedIntent
) -> str:
    if not parsed.description:
        return "Tolong sebutkan nama tabungannya ya."
    saving_name = _normalize_saving_name(parsed.description)
    if parsed.amount is None:
        _set_pending_action(
            user_id=user_id,
            action_type="create_saving_target",
            data={"description": saving_name},
        )
        return (
            f"Oke, kita siapin tabungan \"{saving_name}\". "
            "Sekarang sebutkan target nominalnya ya, contoh: 50jt atau 2.5 juta."
        )

    existing = await _find_saving_account(session, user_id, saving_name)
    if existing:
        display_name = _normalize_saving_name(existing.name)
        if display_name != saving_name:
            existing.name = saving_name
        return (
            f"Tabungan \"{display_name}\" sudah ada dengan target "
            f"{_format_currency(float(existing.target_amount), existing.currency)}."
        )

    account = models.SavingsAccount(
        user_id=user_id,
        name=saving_name,
        target_amount=Decimal(str(parsed.amount)),
        current_amount=Decimal("0"),
        currency=parsed.currency or "IDR",
    )
    session.add(account)
    await session.flush()
    return (
        f"✅ Tabungan \"{account.name}\" siap dengan target "
        f"{_format_currency(parsed.amount, account.currency)}. Semangat menabung!"
    )


async def _handle_savings_movement(
    session: AsyncSession, user_id: int, parsed: ParsedIntent
) -> str:
    if not parsed.description:
        return "Tabungan mana yang dimaksud?"
    if parsed.amount is None:
        return "Berapa nominalnya?"

    saving_name = _normalize_saving_name(parsed.description)
    account = await _find_saving_account(session, user_id, saving_name)
    if not account:
        hint = await _format_savings_hint(session, user_id)
        return f"Tabungan tidak ditemukan. Coba cek nama tabungannya.{hint}"

    display_name = _normalize_saving_name(account.name)
    if account.name != display_name:
        account.name = display_name

    amount_decimal = Decimal(str(parsed.amount))
    if parsed.intent == "deposit_saving":
        account.current_amount += amount_decimal
        direction = "deposit"
        message = f"✅ Setoran {_format_currency(parsed.amount, account.currency)} ke tabungan {account.name}."
    else:
        if account.current_amount - amount_decimal < Decimal("0"):
            return "Saldo tabungan tidak mencukupi."
        account.current_amount -= amount_decimal
        direction = "withdraw"
        message = f"✅ Penarikan {_format_currency(parsed.amount, account.currency)} dari tabungan {account.name}."

    movement = models.SavingsTransaction(
        savings_account_id=account.id,
        user_id=user_id,
        direction=direction,
        amount=amount_decimal,
        note=parsed.description,
        source="whatsapp",
    )
    session.add(movement)
    await session.flush()
    return message


async def _handle_list_savings(session: AsyncSession, user_id: int) -> str:
    accounts = await _get_savings_accounts(session, user_id)
    if not accounts:
        return "Belum ada tabungan yang tercatat. Coba buat tabungan baru dengan mengetik \"Bikin tabungan <nama>\"."
    grouped = {}
    order = []
    for account in accounts:
        key = _normalize_saving_name(account.name).lower()
        if key not in grouped:
            order.append(key)
            grouped[key] = {
                "name": _normalize_saving_name(account.name),
                "currency": account.currency or "IDR",
                "target": float(account.target_amount or 0),
                "current": float(account.current_amount or 0),
            }
        else:
            grouped[key]["current"] += float(account.current_amount or 0)
            grouped[key]["target"] = max(
                grouped[key]["target"], float(account.target_amount or 0)
            )

    total_target = sum(item["target"] for item in grouped.values())
    total_current = sum(item["current"] for item in grouped.values())
    currency = next(iter(grouped.values()))["currency"]
    overall_progress = 0.0
    if total_target > 0:
        overall_progress = min((total_current / total_target) * 100, 999.0)

    lines = [
        "🏦 *Tabungan Aktifmu*",
        f"Total terkumpul {_format_currency(total_current, currency)} dari {_format_currency(total_target, currency)} ({overall_progress:.0f}% tercapai)",
        _build_progress_bar(overall_progress),
        "",
    ]

    for idx, key in enumerate(order, start=1):
        info = grouped[key]
        target = info["target"]
        current = info["current"]
        currency = info["currency"]
        progress = 0.0
        if target > 0:
            progress = min((current / target) * 100, 999.0)
        remaining = max(target - current, 0)

        lines.append(f"{idx}. {info['name']}")
        lines.append(f"   🎯 Target: {_format_currency(target, currency)}")
        lines.append(f"   💰 Terkumpul: {_format_currency(current, currency)} ({progress:.0f}% tercapai)")
        lines.append("   " + _build_progress_bar(progress))
        if remaining > 0:
            lines.append(f"   📉 Sisa {_format_currency(remaining, currency)} lagi")
        else:
            lines.append("   ✅ Target sudah tercapai! 🎉")
        lines.append("")

    lines.append("💡 Gunakan \"Setor <nominal> ke tabungan <nama>\" atau \"Tarik ...\" untuk memperbarui.")
    return "\n".join(lines)


async def _handle_list_categories(session: AsyncSession, user_id: int) -> str:
    stmt = (
        select(
            models.Category,
            func.count(models.Transaction.id),
            func.coalesce(
                func.sum(
                    case((models.Transaction.direction == "income", models.Transaction.amount), else_=0)
                ),
                0,
            ),
            func.coalesce(
                func.sum(
                    case((models.Transaction.direction == "expense", models.Transaction.amount), else_=0)
                ),
                0,
            ),
        )
        .join(models.Transaction, models.Transaction.category_id == models.Category.id, isouter=True)
        .where(models.Category.user_id == user_id)
        .group_by(models.Category.id)
        .order_by(models.Category.name.asc())
    )
    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return (
            "Belum ada kategori khusus. Coba tambah dengan mengetik "
            "\"tambah kategori <nama>\"."
        )

    lines = ["🏷️ *Kategori Aktif*"]
    total_income = 0.0
    total_expense = 0.0

    for idx, (category, tx_count, income_sum, expense_sum) in enumerate(rows, start=1):
        total_income += float(income_sum or 0)
        total_expense += float(expense_sum or 0)
        lines.append(f"{idx}. {category.name}")
        lines.append(f"   • Transaksi: {tx_count}")
        if income_sum:
            lines.append(
                f"   • Pemasukan: {_format_currency(float(income_sum), 'IDR')}"
            )
        if expense_sum:
            lines.append(
                f"   • Pengeluaran: {_format_currency(float(expense_sum), 'IDR')}"
            )
        lines.append("")

    balance = total_income - total_expense
    balance_icon = "💰" if balance >= 0 else "⚠️"
    lines.append(
        f"📥 Total Pemasukan: {_format_currency(total_income, 'IDR')}"
    )
    lines.append(
        f"📤 Total Pengeluaran: {_format_currency(total_expense, 'IDR')}"
    )
    lines.append(f"{balance_icon} Saldo Bersih: {_format_currency(balance, 'IDR')}")
    lines.append("")
    lines.append(
        "💡 Contoh perintah: 'tambah kategori Transport', 'ubah kategori Jajan menjadi Hiburan', atau 'hapus kategori Belanja'."
    )

    return "\n".join(lines)


async def _handle_add_category(session: AsyncSession, user_id: int, text: str | None) -> str:
    name, _ = _extract_category_names(text, "add")
    if not name:
        return "Sebutkan nama kategori yang ingin dibuat, misalnya 'tambah kategori Transport'."

    existing = await _find_category_by_name(session, user_id, name)
    if existing:
        return f"Kategori \"{existing.name}\" sudah ada."

    category = models.Category(user_id=user_id, name=name)
    session.add(category)
    await session.commit()
    return f"✅ Kategori \"{name}\" berhasil ditambahkan."


async def _handle_rename_category(session: AsyncSession, user_id: int, text: str | None) -> str:
    old_name, new_name = _extract_category_names(text, "rename")
    if not old_name or not new_name:
        return "Gunakan format 'ubah kategori Lama menjadi Baru'."

    category = await _find_category_by_name(session, user_id, old_name)
    if not category:
        hint = await _format_categories_hint(session, user_id)
        return f"Kategori \"{old_name}\" tidak ditemukan.{hint}"

    duplicate = await _find_category_by_name(session, user_id, new_name)
    if duplicate and duplicate.id != category.id:
        return f"Kategori \"{new_name}\" sudah tersedia."

    category.name = new_name
    await session.commit()
    return f"✅ Kategori \"{old_name}\" diganti menjadi \"{new_name}\"."


async def _handle_delete_category(session: AsyncSession, user_id: int, text: str | None) -> str:
    name, _ = _extract_category_names(text, "delete")
    if not name:
        return "Sebutkan kategori yang ingin dihapus, contohnya 'hapus kategori Transport'."

    category = await _find_category_by_name(session, user_id, name)
    if not category:
        hint = await _format_categories_hint(session, user_id)
        return f"Kategori \"{name}\" tidak ditemukan.{hint}"

    await session.delete(category)
    await session.commit()
    return f"✅ Kategori \"{name}\" berhasil dihapus. Semua transaksi terkait kini tanpa kategori."


async def _handle_transactions_request(
    session: AsyncSession, user_id: int, payload: IncomingMessage
) -> str:
    window = _determine_report_window(payload)
    text_lower = (payload.text or "").lower()
    if "transaksi" in text_lower:
        specific_keywords = [
            "hari",
            "minggu",
            "tahun",
            "kemarin",
            "lalu",
            "bulan",
            "tanggal",
            "/",
            "-",
        ]
        if window.title == "Hari Ini" and not any(keyword in text_lower for keyword in specific_keywords):
            tz = ZoneInfo(settings.timezone)
            base = (payload.timestamp or datetime.now(timezone.utc)).astimezone(tz)
            start_local = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_day = calendar.monthrange(base.year, base.month)[1]
            end_local = base.replace(
                day=last_day,
                hour=23,
                minute=59,
                second=59,
                microsecond=999999,
            )
            window = ReportWindow(
                title="Bulan Ini",
                start=start_local,
                end=end_local,
                period_label=_format_period(start_local, end_local, tz),
            )

    start_utc = window.start.astimezone(timezone.utc)
    end_utc = window.end.astimezone(timezone.utc)

    stmt = (
        select(models.Transaction, models.Category.name)
        .join(models.Category, models.Transaction.category_id == models.Category.id, isouter=True)
        .where(
            models.Transaction.user_id == user_id,
            models.Transaction.tx_datetime >= start_utc,
            models.Transaction.tx_datetime <= end_utc,
        )
        .order_by(models.Transaction.tx_datetime.desc())
    )

    result = await session.execute(stmt)
    rows = result.all()
    if not rows:
        return (
            f"📒 *Daftar Transaksi — {window.title}*\n"
            f"🗓️ {window.period_label}\n"
            "Belum ada transaksi pada periode ini."
        )

    tz = ZoneInfo(settings.timezone)
    grouped: dict[datetime.date, list[tuple[datetime, models.Transaction, str | None]]] = defaultdict(list)
    order: list[datetime.date] = []
    total_income = 0.0
    total_expense = 0.0
    income_count = 0
    expense_count = 0

    for tx, category_name in rows:
        local_dt = tx.tx_datetime.astimezone(tz)
        date_key = local_dt.date()
        if date_key not in grouped:
            order.append(date_key)
        grouped[date_key].append((local_dt, tx, category_name))

        amount_float = float(tx.amount)
        if tx.direction == "income":
            total_income += amount_float
            income_count += 1
        elif tx.direction == "expense":
            total_expense += amount_float
            expense_count += 1

    lines = [
        f"📒 *Daftar Transaksi — {window.title}*",
        f"🗓️ {window.period_label}",
        "",
    ]

    for date_key in order:
        date_label = date_key.strftime("%A, %d %b %Y")
        lines.append(f"📆 {date_label}")
        for local_dt, tx, category_name in grouped[date_key]:
            time_label = local_dt.strftime("%H:%M")
            amount_fmt = _format_currency(float(tx.amount), tx.currency)
            direction_icon = "📤" if tx.direction == "expense" else "📥"
            description = tx.description or "(tanpa deskripsi)"
            category_label = category_name or "Tanpa kategori"
            lines.append(
                f"  • [{time_label}] {direction_icon} {amount_fmt} — {description} ({category_label})"
            )
        lines.append("")

    balance = total_income - total_expense
    balance_icon = "💰" if balance >= 0 else "⚠️"
    lines.append(
        f"📥 Total Pemasukan: {_format_currency(total_income, 'IDR')} ({income_count} transaksi)"
    )
    lines.append(
        f"📤 Total Pengeluaran: {_format_currency(total_expense, 'IDR')} ({expense_count} transaksi)"
    )
    lines.append(f"{balance_icon} Saldo Bersih: {_format_currency(balance, 'IDR')}")
    lines.append("")
    lines.append(
        "💡 Gunakan 'daftar transaksi minggu ini' atau 'daftar transaksi bulan lalu' untuk periode lain."
    )

    return "\n".join(lines)


async def _handle_balance_request(session: AsyncSession, user_id: int) -> str:
    income_stmt = select(func.coalesce(func.sum(models.Transaction.amount), 0)).where(
        models.Transaction.user_id == user_id,
        models.Transaction.direction == "income",
    )
    expense_stmt = select(func.coalesce(func.sum(models.Transaction.amount), 0)).where(
        models.Transaction.user_id == user_id,
        models.Transaction.direction == "expense",
    )
    income_count_stmt = select(func.count(models.Transaction.id)).where(
        models.Transaction.user_id == user_id,
        models.Transaction.direction == "income",
    )
    expense_count_stmt = select(func.count(models.Transaction.id)).where(
        models.Transaction.user_id == user_id,
        models.Transaction.direction == "expense",
    )

    income_total = (await session.execute(income_stmt)).scalar() or 0
    expense_total = (await session.execute(expense_stmt)).scalar() or 0
    income_count = (await session.execute(income_count_stmt)).scalar() or 0
    expense_count = (await session.execute(expense_count_stmt)).scalar() or 0

    balance = float(income_total) - float(expense_total)
    balance_icon = "💰" if balance >= 0 else "⚠️"

    lines = [
        "💼 *Saldo Saat Ini*",
        f"📥 Pemasukan: {_format_currency(income_total, 'IDR')} ({income_count} transaksi)",
        f"📤 Pengeluaran: {_format_currency(expense_total, 'IDR')} ({expense_count} transaksi)",
        f"{balance_icon} Saldo Bersih: {_format_currency(balance, 'IDR')}",
        "",
        "💡 Kamu bisa minta 'laporan bulan ini' atau 'lihat tabungan' untuk detail lainnya.",
    ]
    return "\n".join(lines)


async def _handle_report_request(
    session: AsyncSession, user_id: int, payload: IncomingMessage
) -> str:
    window = _determine_report_window(payload)
    start_utc = window.start.astimezone(timezone.utc)
    end_utc = window.end.astimezone(timezone.utc)

    tx_stmt = select(models.Transaction).where(
        models.Transaction.user_id == user_id,
        models.Transaction.tx_datetime >= start_utc,
        models.Transaction.tx_datetime <= end_utc,
    )
    tx_result = await session.execute(tx_stmt)
    transactions = tx_result.scalars().all()

    if not transactions:
        return (
            f"📊 *Ringkasan Keuangan — {window.title}*\n"
            f"🗓️ {window.period_label}\n"
            "Belum ada transaksi yang tercatat pada periode ini. "
            "Coba catat transaksi baru atau minta rentang lain, misalnya \"laporan bulan lalu\"."
        )

    income_txs = [t for t in transactions if t.direction == "income"]
    expense_txs = [t for t in transactions if t.direction == "expense"]

    total_income = sum(float(t.amount) for t in income_txs)
    total_expense = sum(float(t.amount) for t in expense_txs)
    balance = total_income - total_expense

    lines = [
        f"📊 *Ringkasan Keuangan — {window.title}*",
        f"🗓️ {window.period_label}",
        f"📥 Pemasukan • {_format_currency(total_income, 'IDR')} ({len(income_txs)} transaksi)",
        f"📤 Pengeluaran • {_format_currency(total_expense, 'IDR')} ({len(expense_txs)} transaksi)",
    ]
    balance_icon = "💰" if balance >= 0 else "⚠️"
    lines.append(f"{balance_icon} Saldo Bersih • {_format_currency(balance, 'IDR')}")

    top_expense = await _fetch_top_categories(
        session,
        user_id=user_id,
        direction="expense",
        start=start_utc,
        end=end_utc,
        limit=3,
    )
    if top_expense:
        lines.append("🏷️ Pengeluaran Terbesar:")
        for idx, (name, amount) in enumerate(top_expense, start=1):
            label = name or "Tanpa kategori"
            lines.append(f"  {idx}. {label} — {_format_currency(amount, 'IDR')}")

    top_income = await _fetch_top_categories(
        session,
        user_id=user_id,
        direction="income",
        start=start_utc,
        end=end_utc,
        limit=3,
    )
    if top_income:
        lines.append("🏷️ Sumber Pemasukan Terbesar:")
        for idx, (name, amount) in enumerate(top_income, start=1):
            label = name or "Tanpa kategori"
            lines.append(f"  {idx}. {label} — {_format_currency(amount, 'IDR')}")

    lines.append("🧠 Tips: kamu bisa minta \"laporan minggu ini\", \"laporan bulan lalu\", atau sebut tanggal tertentu.")
    return "\n".join(lines)


@dataclass
class ReportWindow:
    title: str
    start: datetime
    end: datetime
    period_label: str


def _determine_report_window(payload: IncomingMessage) -> ReportWindow:
    tz = ZoneInfo(settings.timezone)
    base = (payload.timestamp or datetime.now(timezone.utc)).astimezone(tz)
    text = (payload.text or "").lower()

    start_local = base.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = base.replace(hour=23, minute=59, second=59, microsecond=999999)
    title = "Hari Ini"

    def make_window(start: datetime, end: datetime, title_label: str) -> ReportWindow:
        return ReportWindow(
            title=title_label,
            start=start,
            end=end,
            period_label=_format_period(start, end, tz),
        )

    if "kemarin" in text:
        day = base - timedelta(days=1)
        return make_window(
            day.replace(hour=0, minute=0, second=0, microsecond=0),
            day.replace(hour=23, minute=59, second=59, microsecond=999999),
            "Kemarin",
        )

    if "minggu lalu" in text:
        current_week_start = base - timedelta(days=base.weekday())
        start = current_week_start - timedelta(days=7)
        end = start + timedelta(days=6)
        return make_window(
            start.replace(hour=0, minute=0, second=0, microsecond=0),
            end.replace(hour=23, minute=59, second=59, microsecond=999999),
            "Minggu Lalu",
        )

    if "minggu ini" in text:
        start = base - timedelta(days=base.weekday())
        end = start + timedelta(days=6)
        return make_window(
            start.replace(hour=0, minute=0, second=0, microsecond=0),
            end.replace(hour=23, minute=59, second=59, microsecond=999999),
            "Minggu Ini",
        )

    if "bulan lalu" in text:
        year = base.year
        month = base.month - 1
        if month == 0:
            month = 12
            year -= 1
        start = base.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day = calendar.monthrange(year, month)[1]
        end = base.replace(year=year, month=month, day=last_day, hour=23, minute=59, second=59, microsecond=999999)
        return make_window(start, end, "Bulan Lalu")

    if "bulan ini" in text:
        start = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day = calendar.monthrange(base.year, base.month)[1]
        end = base.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)
        return make_window(start, end, "Bulan Ini")

    if "tahun lalu" in text:
        start = base.replace(year=base.year - 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = base.replace(year=base.year - 1, month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
        return make_window(start, end, "Tahun Lalu")

    if "tahun ini" in text:
        start = base.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = base.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
        return make_window(start, end, "Tahun Ini")

    date_match = re.search(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", text)
    if date_match:
        day, month, year = date_match.groups()
        day = int(day)
        month = int(month)
        if year:
            year = int(year)
            if year < 100:
                year += 2000
        else:
            year = base.year
        try:
            specific = datetime(year, month, day, tzinfo=tz)
            return make_window(
                specific.replace(hour=0, minute=0, second=0, microsecond=0),
                specific.replace(hour=23, minute=59, second=59, microsecond=999999),
                specific.strftime("%d %B %Y"),
            )
        except ValueError:
            pass

    if "hari ini" in text or "today" in text or "laporan" in text:
        return make_window(start_local, end_local, "Hari Ini")

    return make_window(start_local, end_local, title)


def _format_period(start: datetime, end: datetime, tz: ZoneInfo) -> str:
    start_local = start.astimezone(tz)
    end_local = end.astimezone(tz)
    if start_local.date() == end_local.date():
        return start_local.strftime("%d %b %Y")
    return f"{start_local.strftime('%d %b %Y')} – {end_local.strftime('%d %b %Y')}"


async def _fetch_top_categories(
    session: AsyncSession,
    *,
    user_id: int,
    direction: str,
    start: datetime,
    end: datetime,
    limit: int,
) -> list[tuple[str | None, float]]:
    stmt = (
        select(models.Category.name, func.sum(models.Transaction.amount))
        .join(models.Category, models.Transaction.category_id == models.Category.id, isouter=True)
        .where(
            models.Transaction.user_id == user_id,
            models.Transaction.direction == direction,
            models.Transaction.tx_datetime >= start,
            models.Transaction.tx_datetime <= end,
        )
        .group_by(models.Category.name)
        .order_by(func.sum(models.Transaction.amount).desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    records: list[tuple[str | None, float]] = []
    for name, total in result.all():
        if total is not None:
            records.append((name, float(total)))
    return records


async def _get_savings_accounts(session: AsyncSession, user_id: int) -> list[models.SavingsAccount]:
    stmt = (
        select(models.SavingsAccount)
        .where(models.SavingsAccount.user_id == user_id)
        .order_by(models.SavingsAccount.created_at.asc())
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def _find_saving_account(
    session: AsyncSession, user_id: int, name: str
) -> models.SavingsAccount | None:
    normalized = _normalize_saving_name(name).lower()
    accounts = await _get_savings_accounts(session, user_id)
    for account in accounts:
        if _normalize_saving_name(account.name).lower() == normalized:
            return account
    return None


async def _format_savings_hint(session: AsyncSession, user_id: int) -> str:
    accounts = await _get_savings_accounts(session, user_id)
    if not accounts:
        return ""
    names = list({
        _normalize_saving_name(acc.name)
        for acc in accounts
    })
    preview = ", ".join(names[:3])
    if len(names) > 3:
        preview += ", ..."
    return f" Coba gunakan salah satu nama ini: {preview}."


async def _get_categories(session: AsyncSession, user_id: int) -> list[models.Category]:
    stmt = (
        select(models.Category)
        .where(models.Category.user_id == user_id)
        .order_by(models.Category.name.asc())
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def _find_category_by_name(
    session: AsyncSession, user_id: int, name: str
) -> models.Category | None:
    normalized = _normalize_category_name(name).lower()
    stmt = select(models.Category).where(
        models.Category.user_id == user_id,
        func.lower(models.Category.name) == normalized,
    )
    return await session.scalar(stmt)


async def _format_categories_hint(session: AsyncSession, user_id: int) -> str:
    categories = await _get_categories(session, user_id)
    if not categories:
        return ""
    names = [category.name for category in categories[:5]]
    preview = ", ".join(names)
    if len(categories) > 5:
        preview += ", ..."
    return f" Coba gunakan salah satu nama kategori: {preview}."


def _looks_like_amount_only(text: str | None) -> bool:
    if not text:
        return False
    sanitized = text.strip().lower()
    if not sanitized:
        return False
    sanitized = sanitized.replace("rp", "")
    sanitized = sanitized.replace(" ", "")
    sanitized = sanitized.replace(".", "")
    sanitized = sanitized.replace(",", "")
    sanitized = sanitized.replace("-", "")
    pattern = r"^\d+(?:rb|ribu|k|jt|juta|m|miliar|milyar)?$"
    return re.fullmatch(pattern, sanitized) is not None


def _build_progress_bar(percent: float) -> str:
    total_blocks = 10
    capped = max(0.0, min(percent or 0.0, 100.0))
    filled = int(round((capped / 100.0) * total_blocks))
    filled = max(0, min(filled, total_blocks))
    bar = "█" * filled + "░" * (total_blocks - filled)
    return f"[{bar}] {capped:.0f}%"


def _text_mentions_saving(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    keywords = [
        "tabungan",
        "nabung",
        "simpan uang",
        "saving",
        "tambah tabung",
        "deposit tabungan",
    ]
    return any(keyword in lowered for keyword in keywords)


def _text_requests_savings_list(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    keywords = [
        "lihat tabungan",
        "cek tabungan",
        "daftar tabungan",
        "tabungan saya",
        "status tabungan",
        "saldo tabungan",
        "list tabungan",
    ]
    return any(keyword in lowered for keyword in keywords)


def _text_requests_balance(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower().strip()
    keywords = [
        "saldo",
        "cek saldo",
        "lihat saldo",
        "sisa uang",
        "balance",
        "total uang",
    ]
    return any(keyword in lowered for keyword in keywords)


def _text_requests_transactions_list(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    keywords = [
        "daftar transaksi",
        "lihat transaksi",
        "riwayat transaksi",
        "history transaksi",
        "cek transaksi",
    ]
    return any(keyword in lowered for keyword in keywords)


def _text_requests_category_list(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    keywords = [
        "daftar kategori",
        "lihat kategori",
        "cek kategori",
    ]
    return any(keyword in lowered for keyword in keywords)


def _text_requests_category_add(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    keywords = [
        "tambah kategori",
        "kategori baru",
        "buat kategori",
    ]
    return any(keyword in lowered for keyword in keywords)


def _text_requests_category_delete(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    keywords = [
        "hapus kategori",
        "delete kategori",
        "remove kategori",
    ]
    return any(keyword in lowered for keyword in keywords)


def _text_requests_category_rename(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    keywords = [
        "ubah kategori",
        "rename kategori",
        "ganti kategori",
    ]
    return any(keyword in lowered for keyword in keywords)


def _normalize_saving_name(raw: str) -> str:
    candidate = (raw or "").strip()
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = re.sub(r"[\"'`]+", "", candidate)
    if not candidate:
        return "Tabungan Baru"

    tokens = candidate.split()
    leading_stopwords = {
        "bikin",
        "buat",
        "ayo",
        "mau",
        "ingin",
        "pengen",
        "tolong",
        "tambahkan",
        "buatkan",
        "please",
        "tolongi",
        "buatin",
    }
    filtered: list[str] = []
    skip_leading = True
    for word in tokens:
        lower = word.lower()
        if skip_leading and lower in leading_stopwords:
            continue
        if skip_leading and lower == "tabungan":
            continue
        skip_leading = False
        filtered.append(word)

    if not filtered:
        filtered = tokens

    name = " ".join(filtered).strip()
    if not name:
        name = "Tabungan Baru"
    else:
        name = name.title()
    return name


def _extract_saving_reference(text: str | None) -> str | None:
    if not text:
        return None

    match = re.search(r"tabungan\s+([a-z0-9\s]+)", text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(r"[\.,!?]+$", "", candidate)
        if candidate:
            return _normalize_saving_name(candidate)

    match = re.search(r"ke\s+([a-z0-9\s]+)", text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(r"[\.,!?]+$", "", candidate)
        if candidate:
            return _normalize_saving_name(candidate)

    return None


def _normalize_category_name(raw: str) -> str:
    candidate = (raw or "").strip()
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = re.sub(r"[\"'`]+", "", candidate)
    if not candidate:
        return "Umum"

    tokens = candidate.split()
    if tokens and tokens[0].lower() == "kategori":
        tokens = tokens[1:]
    name = " ".join(tokens).strip()
    if not name:
        name = candidate
    return name.title()


def _extract_category_names(text: str | None, action: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None

    original = text.strip()

    def slice_after(keyword: str) -> str | None:
        lowered = original.lower()
        idx = lowered.find(keyword)
        if idx == -1:
            return None
        value = original[idx + len(keyword):].strip(" :")
        return value or None

    if action == "add":
        for keyword in ["tambah kategori", "kategori baru", "buat kategori"]:
            name = slice_after(keyword)
            if name:
                return _normalize_category_name(name), None
        return None, None

    if action == "delete":
        for keyword in ["hapus kategori", "delete kategori", "remove kategori"]:
            name = slice_after(keyword)
            if name:
                return _normalize_category_name(name), None
        return None, None

    if action == "rename":
        patterns = [
            r"ubah\s+kategori\s+(.+?)\s+(?:menjadi|ke)\s+(.+)",
            r"ganti\s+kategori\s+(.+?)\s+(?:menjadi|ke)\s+(.+)",
            r"rename\s+kategori\s+(.+?)\s+(?:menjadi|ke)\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, original, flags=re.IGNORECASE)
            if match:
                old_name = _normalize_category_name(match.group(1))
                new_name = _normalize_category_name(match.group(2))
                return old_name, new_name
        return None, None

    return None, None


def _message_looks_like_transaction(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if (
        _text_requests_savings_list(text)
        or _text_requests_balance(text)
        or _text_requests_transactions_list(text)
        or _text_requests_category_list(text)
        or _text_requests_category_add(text)
        or _text_requests_category_delete(text)
        or _text_requests_category_rename(text)
    ):
        return False
    if re.search(r"\d", text):
        return True
    transaction_keywords = [
        "beli",
        "bayar",
        "belanja",
        "makan",
        "ongkir",
        "tagihan",
        "top up",
        "transfer",
        "gaji",
        "terima",
        "masuk",
        "setor",
        "tarik",
        "deposit",
        "withdraw",
        "pemasukan",
        "pengeluaran",
    ]
    return any(keyword in lowered for keyword in transaction_keywords)


def _set_pending_action(*, user_id: int, action_type: str, data: dict[str, Any]) -> None:
    expires = datetime.now(timezone.utc) + timedelta(minutes=5)
    _pending_actions[user_id] = PendingAction(action_type=action_type, data=data, expires_at=expires)


def _get_pending_action(user_id: int) -> PendingAction | None:
    action = _pending_actions.get(user_id)
    if not action:
        return None
    if action.expires_at < datetime.now(timezone.utc):
        _pending_actions.pop(user_id, None)
        return None
    return action


def _clear_pending_action(user_id: int) -> None:
    _pending_actions.pop(user_id, None)


async def _maybe_handle_pending_action(
    session: AsyncSession,
    user_id: int,
    payload: IncomingMessage,
) -> PendingActionResult | None:
    action = _get_pending_action(user_id)
    if not action:
        return None

    text = (payload.text or "").strip()
    lowered = text.lower()

    cancel_keywords = {
        "batal",
        "cancel",
        "ga jadi",
        "gajadi",
        "tidak jadi",
        "udah",
        "sudah",
        "apaan sih",
        "apa sih",
    }
    if any(keyword in lowered for keyword in cancel_keywords):
        _clear_pending_action(user_id)
        return PendingActionResult(
            reply="Baik, aku batalkan permintaan sebelumnya. Ada yang bisa kubantu lagi?",
            intent="smalltalk",
        )

    if (
        _text_requests_savings_list(text)
        or _text_requests_balance(text)
        or _text_requests_transactions_list(text)
        or _text_requests_category_list(text)
        or _text_requests_category_add(text)
        or _text_requests_category_delete(text)
        or _text_requests_category_rename(text)
    ):
        _clear_pending_action(user_id)
        return None

    if action.action_type == "create_saving_target":
        amount = _extract_amount_indonesia(text)
        description = action.data.get("description", "tabungan")
        if amount is None:
            return PendingActionResult(
                reply=f"Aku masih menunggu target untuk tabungan \"{description}\". Contoh: 500 ribu atau 1,5jt.",
                intent="create_saving",
            )

        parsed = ParsedIntent(
            intent="create_saving",
            amount=amount,
            description=description,
            currency="IDR",
        )
        _clear_pending_action(user_id)
        reply = await _handle_create_saving(session, user_id, parsed)
        return PendingActionResult(reply=reply, intent="create_saving")

    if action.action_type == "create_transaction_details":
        stored_amount = action.data.get("amount")
        stored_currency = action.data.get("currency") or "IDR"

        if not text or not any(ch.isalpha() for ch in text):
            if stored_amount is not None:
                amount_fmt = _format_currency(float(stored_amount), stored_currency)
                return PendingActionResult(
                    reply=(
                        f"Nominal {amount_fmt} sudah ada. Ceritakan ini untuk apa dan apakah pemasukan atau pengeluaran, ya."
                    ),
                    intent="create_transaction",
                )
            return PendingActionResult(
                reply="Bisa jelaskan ini transaksi apa dan berapa nominalnya?",
                intent="create_transaction",
            )

        parsed, audit_payload, model_name = await _parse_intent_with_ai(text)
        if parsed.amount is None and stored_amount is not None:
            parsed.amount = float(stored_amount)
        if parsed.amount is None:
            return PendingActionResult(
                reply="Masih belum ada nominal yang jelas. Contoh: 'beli pulsa 50rb' atau 'terima gaji 2jt'.",
                intent="create_transaction",
            )

        parsed.currency = parsed.currency or stored_currency
        parsed.description = parsed.description or text
        if parsed.intent in {"create_saving", "deposit_saving", "withdraw_saving"}:
            reference = _extract_saving_reference(text)
            if reference:
                parsed.description = reference
        parsed.direction = parsed.direction or _heuristic_direction(text.lower()) or "expense"

        _clear_pending_action(user_id)
        reply = await _handle_transaction_intent(session, user_id, payload, parsed, text)
        await _log_ai_audit(session, user_id, text, audit_payload, model_name, success=True)
        return PendingActionResult(reply=reply, intent="create_transaction")

    return None


async def _resolve_category(
    session: AsyncSession, user_id: int, suggestion: str | None
) -> int | None:
    if not suggestion:
        return None
    stmt = select(models.Category).where(
        models.Category.user_id == user_id,
        func.lower(models.Category.name) == suggestion.lower(),
    )
    category = await session.scalar(stmt)
    if category:
        return category.id

    category = models.Category(user_id=user_id, name=suggestion)
    session.add(category)
    await session.flush()
    return category.id


async def _get_category_name(session: AsyncSession, category_id: int | None) -> str | None:
    if category_id is None:
        return None
    stmt = select(models.Category.name).where(models.Category.id == category_id)
    return await session.scalar(stmt)


async def _build_transaction_summary(
    *,
    session: AsyncSession,
    user_id: int,
    transaction: models.Transaction,
    category_name: str | None,
) -> str:
    tz = ZoneInfo(settings.timezone)
    tx_local = transaction.tx_datetime.astimezone(tz)
    timestamp_str = tx_local.strftime("%d %b %Y • %H:%M")
    tz_label = tx_local.tzname() or settings.timezone
    amount_fmt = _format_currency(float(transaction.amount), transaction.currency)
    direction_phrase = "Pemasukan" if transaction.direction == "income" else "Pengeluaran"
    trend_icon = "📥" if transaction.direction == "income" else "📤"

    lines = [
        "✨ *Catatan Keuangan Tersimpan!*",
        f"📆 {timestamp_str} {tz_label}",
        f"{trend_icon} {direction_phrase} • {amount_fmt}",
    ]

    if category_name:
        lines.append(f"🏷️ Kategori • {category_name}")
    else:
        lines.append("🏷️ Kategori • (Belum ditetapkan)")

    day_start_local = tx_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end_local = tx_local.replace(hour=23, minute=59, second=59, microsecond=999999)
    today_total = await _sum_amount(
        session,
        user_id=user_id,
        direction=transaction.direction,
        start=day_start_local.astimezone(timezone.utc),
        end=day_end_local.astimezone(timezone.utc),
        category_id=transaction.category_id,
    )
    month_total = await _sum_amount_month(
        session,
        user_id=user_id,
        direction=transaction.direction,
        tx_datetime=transaction.tx_datetime,
        category_id=transaction.category_id,
    )

    summary_parts = []
    if today_total is not None:
        daily_caption = "masuk" if transaction.direction == "income" else "keluar"
        summary_parts.append(
            f"Hari ini {daily_caption} {_format_currency(today_total, transaction.currency)}"
        )
    if month_total is not None:
        monthly_caption = "masuk" if transaction.direction == "income" else "keluar"
        summary_parts.append(
            f"Bulan ini {monthly_caption} {_format_currency(month_total, transaction.currency)}"
        )

    if summary_parts:
        if transaction.category_id and category_name:
            label = f"Kategori {category_name}"
        else:
            label = f"Total {direction_phrase}"
        lines.append("📊 " + label + " • " + " | ".join(summary_parts))

    if transaction.description:
        lines.append(f"📝 Catatan • {transaction.description}")

    return "\n".join(lines)


async def _log_wa_message(
    session: AsyncSession,
    *,
    user_id: int | None,
    wa_from: str,
    body: str,
    direction: str,
    intent: str | None = None,
    response: str | None = None,
) -> None:
    message = models.WAMessage(
        user_id=user_id,
        wa_from=wa_from,
        wa_body=body,
        direction=direction,
        intent=intent,
        response=response,
    )
    session.add(message)
    await session.flush()


async def _log_ai_audit(
    session: AsyncSession,
    user_id: int,
    raw_input: str,
    raw_output: dict[str, object],
    model_name: str,
    success: bool,
) -> None:
    audit = models.AIAudit(
        user_id=user_id,
        model_name=model_name,
        raw_input=raw_input,
        raw_output=json.dumps(raw_output, ensure_ascii=False),
        success=success,
    )
    session.add(audit)
    await session.flush()


async def _call_ai_service(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
    url = f"{settings.ai_service_url}{endpoint}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def _format_ocr_response(ocr_response: dict[str, object]) -> str:
    """Format OCR response menjadi text yang bisa di-parse oleh NLU.
    
    Enhanced untuk handle response dari OCR processor yang lebih lengkap.
    """
    parts = []
    
    # Merchant info
    merchant = ocr_response.get("merchant")
    if merchant:
        parts.append(f"struk dari {merchant}")
    
    receipt_type = ocr_response.get("receipt_type")
    if receipt_type and receipt_type != "unknown":
        parts.append(f"tipe {receipt_type}")
    
    # Amount info - prioritas: total > subtotal
    total = ocr_response.get("total")
    if total:
        parts.append(f"total {total}")
    else:
        subtotal = ocr_response.get("subtotal")
        if subtotal:
            parts.append(f"subtotal {subtotal}")
    
    # Tax & discount
    tax = ocr_response.get("tax")
    if tax:
        parts.append(f"pajak {tax}")
    
    discount = ocr_response.get("discount")
    if discount:
        parts.append(f"diskon {discount}")
    
    # Date & time
    date = ocr_response.get("date")
    if date:
        parts.append(f"tanggal {date}")
    
    time = ocr_response.get("time")
    if time:
        parts.append(f"jam {time}")
    
    # Payment method
    payment = ocr_response.get("payment_method")
    if payment:
        parts.append(f"bayar {payment}")
    
    # Items summary (if available)
    items = ocr_response.get("items")
    if items and isinstance(items, list) and len(items) > 0:
        item_names = [item.get("name", "") for item in items[:3] if item.get("name")]
        if item_names:
            parts.append(f"item: {', '.join(item_names)}")
    
    # Confidence info for logging
    confidence = ocr_response.get("confidence", 0)
    if confidence and confidence < 0.5:
        logger.warning(
            "Low OCR confidence",
            confidence=confidence,
            merchant=merchant,
            total=total,
        )
    
    # Fallback to raw_text if no structured data
    if not parts:
        raw_text = ocr_response.get("raw_text", "")
        if raw_text:
            # Clean up raw text - take first 500 chars
            cleaned = " ".join(raw_text.split())[:500]
            return cleaned
        return ""
    
    return " ".join(parts)


def _heuristic_parse(text: str) -> ParsedIntent:
    lowered = text.lower()
    intent = "create_transaction"
    transactions_keywords = [
        "daftar transaksi",
        "lihat transaksi",
        "cek transaksi",
        "riwayat transaksi",
        "history transaksi",
    ]
    savings_list_keywords = [
        "lihat tabungan",
        "cek tabungan",
        "daftar tabungan",
        "tabungan saya",
        "status tabungan",
        "saldo tabungan",
    ]
    category_list_keywords = [
        "daftar kategori",
        "lihat kategori",
        "cek kategori",
    ]
    category_add_keywords = [
        "tambah kategori",
        "kategori baru",
        "buat kategori",
    ]
    category_delete_keywords = [
        "hapus kategori",
        "delete kategori",
        "remove kategori",
    ]
    category_rename_keywords = [
        "ubah kategori",
        "rename kategori",
        "ganti kategori",
    ]
    if any(keyword in lowered for keyword in transactions_keywords):
        intent = "list_transactions"
    elif any(keyword in lowered for keyword in savings_list_keywords):
        intent = "list_savings"
    elif any(keyword in lowered for keyword in category_list_keywords):
        intent = "list_categories"
    elif any(keyword in lowered for keyword in category_add_keywords):
        intent = "add_category"
    elif any(keyword in lowered for keyword in category_delete_keywords):
        intent = "delete_category"
    elif any(keyword in lowered for keyword in category_rename_keywords):
        intent = "rename_category"
    elif any(word in lowered for word in ["buat tabungan", "nabung", "tabungan"]):
        intent = "create_saving"
    if any(word in lowered for word in ["setor", "deposit"]):
        intent = "deposit_saving"
    if any(word in lowered for word in ["tarik", "withdraw"]):
        intent = "withdraw_saving"
    if "laporan" in lowered:
        intent = "get_report"

    if intent in {"list_savings", "list_transactions", "list_categories"}:
        return ParsedIntent(intent=intent, amount=None, description=None, category_suggestion=None, direction=None)

    amount = _extract_amount_indonesia(text)
    if amount is None:
        amount_match = re.search(r"([0-9]+[.,0-9]*)", text)
        amount = float(amount_match.group(1).replace(".", "").replace(",", ".")) if amount_match else None

    category = _suggest_category(lowered)
    direction = _heuristic_direction(lowered)
    if direction is None and category in _INCOME_CATEGORIES:
        direction = "income"

    description = text.strip()
    if intent in {"create_saving", "deposit_saving", "withdraw_saving"}:
        reference = _extract_saving_reference(text)
        if reference:
            description = reference
    if intent in {"add_category", "delete_category", "rename_category"}:
        map_action = {
            "add_category": "add",
            "delete_category": "delete",
            "rename_category": "rename",
        }
        name1, name2 = _extract_category_names(text, map_action[intent])
        if intent in {"add_category", "delete_category"} and name1:
            description = name1
        elif intent == "rename_category" and name1 and name2:
            description = f"{name1}|{name2}"

    return ParsedIntent(
        intent=intent,
        amount=amount,
        description=description,
        category_suggestion=category,
        direction=direction,
    )


_AMOUNT_SUFFIX_MULTIPLIERS = {
    "rb": 1000,
    "ribu": 1000,
    "k": 1000,
    "jt": 1_000_000,
    "juta": 1_000_000,
    "m": 1_000_000_000,
    "miliar": 1_000_000_000,
    "milyar": 1_000_000_000,
    "ratus": 100,
    "ratusan": 100,
}

_AMOUNT_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(rb|ribu|k|jt|juta|m|miliar|milyar|ratus|ratusan)?",
    re.IGNORECASE,
)

_CATEGORY_KEYWORDS = {
    "Makan": [
        "makan",
        "sarapan",
        "lunch",
        "dinner",
        "kuliner",
        "resto",
        "restaurant",
        "warteg",
        "kopi",
        "coffee",
        "cafe",
        "ayam geprek",
        "nasi",
        "bakso",
    ],
    "Transport": [
        "transport",
        "bensin",
        "bbm",
        "tol",
        "parkir",
        "grab",
        "gojek",
        "go ride",
        "go car",
        "ojek",
        "angkot",
        "kereta",
        "bus",
        "taxi",
        "ongkir",
        "ongkos kirim",
    ],
    "Belanja": [
        "belanja",
        "beli",
        "shopping",
        "mall",
        "baju",
        "celana",
        "sepatu",
        "fashion",
        "thrift",
        "makeup",
        "skincare",
        "kosmetik",
    ],
    "Kesehatan": [
        "dokter",
        "rumah sakit",
        "klinik",
        "obat",
        "vitamin",
        "apotek",
        "bpjs",
        "medical",
    ],
    "Hiburan": [
        "hiburan",
        "bioskop",
        "nonton",
        "game",
        "steam",
        "spotify",
        "netflix",
        "viu",
        "disney",
        "langganan",
        "subscrib",
    ],
    "Tagihan": [
        "listrik",
        "token",
        "pln",
        "pdam",
        "air",
        "internet",
        "wifi",
        "indihome",
        "telkomsel",
        "pulsa",
        "paket data",
        "bayar tagihan",
        "cicilan",
        "angsuran",
    ],
    "Pendidikan": [
        "kuliah",
        "kampus",
        "sekolah",
        "pendidikan",
        "buku",
        "kursus",
        "les",
        "bimbel",
    ],
    "Investasi": [
        "investasi",
        "saham",
        "reksa dana",
        "crypto",
        "emas",
        "bitcoin",
    ],
    "Gaji": [
        "gaji",
        "salary",
        "bayaran",
        "pendapatan",
        "dibayar",
        "transfer gaji",
        "bonus",
        "thr",
        "honor",
    ],
    "Penjualan": [
        "jual",
        "penjualan",
        "invoice",
        "customer",
        "klien",
        "order",
        "pesanan",
        "omzet",
    ],
    "Hadiah": [
        "hadiah",
        "gift",
        "kado",
        "angpao",
        "uang saku",
    ],
}

_CATEGORY_PRIORITY = [
    "Gaji",
    "Penjualan",
    "Hadiah",
    "Investasi",
    "Tagihan",
    "Makan",
    "Belanja",
    "Transport",
    "Kesehatan",
    "Hiburan",
    "Pendidikan",
]

_INCOME_CATEGORIES = {"Gaji", "Penjualan", "Hadiah", "Investasi"}

_INCOME_KEYWORDS = [
    "gaji",
    "terima",
    "masuk",
    "bonus",
    "bayaran",
    "pendapatan",
    "penjualan",
    "omzet",
    "invoice",
    "klien",
    "customer",
    "dapat",
    "thr",
    "honor",
]

_TRANSFER_KEYWORDS = [
    "transfer",
    "kirim",
    "pindah",
    "mutasi",
]

_EXPENSE_KEYWORDS = [
    "bayar",
    "beli",
    "belanja",
    "habis",
    "keluar",
    "biaya",
    "ongkir",
    "ongkos",
    "tagihan",
    "cicilan",
    "angsuran",
]


def _extract_amount_indonesia(text: str) -> float | None:
    candidates: list[tuple[int, float]] = []
    for match in _AMOUNT_PATTERN.finditer(text.lower()):
        number_part, suffix = match.groups()
        if not number_part:
            continue
        normalized = number_part.replace(".", "").replace(",", ".")
        try:
            base_value = float(normalized)
        except ValueError:
            continue
        multiplier = 1.0
        priority = 0
        if suffix:
            suffix_key = suffix.lower()
            multiplier = float(_AMOUNT_SUFFIX_MULTIPLIERS.get(suffix_key, 1))
            priority = 1
        amount = base_value * multiplier
        candidates.append((priority, amount))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return candidates[0][1]


def _suggest_category(lowered_text: str) -> str | None:
    for category in _CATEGORY_PRIORITY:
        keywords = _CATEGORY_KEYWORDS.get(category, [])
        if any(keyword in lowered_text for keyword in keywords):
            return category
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in lowered_text for keyword in keywords):
            return category
    return None


def _heuristic_direction(lowered_text: str) -> str | None:
    if any(word in lowered_text for word in _INCOME_KEYWORDS):
        return "income"
    if any(word in lowered_text for word in _TRANSFER_KEYWORDS):
        return "transfer"
    if any(word in lowered_text for word in _EXPENSE_KEYWORDS):
        return "expense"
    return None


async def _sum_amount(
    session: AsyncSession,
    *,
    user_id: int,
    direction: str,
    start: datetime,
    end: datetime,
    category_id: int | None,
) -> float | None:
    stmt = select(func.sum(models.Transaction.amount)).where(
        models.Transaction.user_id == user_id,
        models.Transaction.direction == direction,
        models.Transaction.tx_datetime >= start,
        models.Transaction.tx_datetime <= end,
    )
    if category_id is not None:
        stmt = stmt.where(models.Transaction.category_id == category_id)

    result = await session.execute(stmt)
    total = result.scalar()
    return float(total) if total is not None else None


async def _sum_amount_month(
    session: AsyncSession,
    *,
    user_id: int,
    direction: str,
    tx_datetime: datetime,
    category_id: int | None,
) -> float | None:
    tz = ZoneInfo(settings.timezone)
    local_dt = tx_datetime.astimezone(tz)
    month_start_local = local_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if local_dt.month == 12:
        next_month_local = month_start_local.replace(year=local_dt.year + 1, month=1)
    else:
        next_month_local = month_start_local.replace(month=local_dt.month + 1)

    month_start = month_start_local.astimezone(timezone.utc)
    next_month = next_month_local.astimezone(timezone.utc)

    stmt = select(func.sum(models.Transaction.amount)).where(
        models.Transaction.user_id == user_id,
        models.Transaction.direction == direction,
        models.Transaction.tx_datetime >= month_start,
        models.Transaction.tx_datetime < next_month,
    )
    if category_id is not None:
        stmt = stmt.where(models.Transaction.category_id == category_id)

    result = await session.execute(stmt)
    total = result.scalar()
    return float(total) if total is not None else None


def _detect_direction(text: str) -> str:
    lowered = text.lower()
    heur = _heuristic_direction(lowered)
    if heur:
        return heur
    return "expense"


def _format_currency(amount: float, currency: str) -> str:
    prefix = "Rp" if currency.upper() == "IDR" else f"{currency.upper()} "
    value = float(amount)
    formatted = f"{value:,.0f}".replace(",", ".")
    return f"{prefix}{formatted}"


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:  # pragma: no cover - invalid formats fallback
            return None
    return None
