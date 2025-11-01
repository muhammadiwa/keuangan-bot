from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal

import httpx
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import models
from app.schemas.wa import IncomingMessage, WAResponse

settings = get_settings()


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

    try:
        text_content = await _extract_text_from_payload(payload)
        logger.debug("Extracted text", text=text_content)

        parsed, audit_payload, model_name = await _parse_intent_with_ai(text_content)

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
        formatted = []
        if ocr_response.get("merchant"):
            formatted.append(f"merchant {ocr_response['merchant']}")
        if ocr_response.get("total"):
            formatted.append(f"total {ocr_response['total']}")
        if ocr_response.get("date"):
            formatted.append(f"tanggal {ocr_response['date']}")
        raw_text = ocr_response.get("raw_text")
        if raw_text:
            formatted.append(raw_text)
        return " ".join(formatted)

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
    if parsed.amount is None:
        return "Aku belum menemukan nominalnya. Tolong kirim ulang dengan jumlah yang jelas."

    direction = parsed.direction or _detect_direction(raw_text)
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

    amount_fmt = _format_currency(parsed.amount, parsed.currency or "IDR")
    category_name = await _get_category_name(session, category_id)
    direction_phrase = "pemasukan" if direction == "income" else "pengeluaran"
    base_message = f"✅ Catat {direction_phrase} {amount_fmt}"
    if category_name:
        base_message += f" kategori {category_name}"
    base_message += ". Balas: ganti <kategori> kalau mau ubah."
    return base_message


async def _handle_create_saving(
    session: AsyncSession, user_id: int, parsed: ParsedIntent
) -> str:
    if not parsed.description:
        return "Tolong sebutkan nama tabungannya ya."
    if parsed.amount is None:
        return "Berapa target tabungan yang ingin dicapai?"

    stmt = select(models.SavingsAccount).where(
        models.SavingsAccount.user_id == user_id,
        func.lower(models.SavingsAccount.name) == parsed.description.lower(),
    )
    existing = await session.scalar(stmt)
    if existing:
        return f"Tabungan {existing.name} sudah ada dengan target Rp{float(existing.target_amount):,.0f}."

    account = models.SavingsAccount(
        user_id=user_id,
        name=parsed.description,
        target_amount=Decimal(str(parsed.amount)),
        current_amount=Decimal("0"),
        currency=parsed.currency or "IDR",
    )
    session.add(account)
    await session.flush()
    return (
        f"✅ Tabungan {account.name} dibuat dengan target "
        f"{_format_currency(parsed.amount, account.currency)}."
    )


async def _handle_savings_movement(
    session: AsyncSession, user_id: int, parsed: ParsedIntent
) -> str:
    if not parsed.description:
        return "Tabungan mana yang dimaksud?"
    if parsed.amount is None:
        return "Berapa nominalnya?"

    stmt = select(models.SavingsAccount).where(
        models.SavingsAccount.user_id == user_id,
        func.lower(models.SavingsAccount.name) == parsed.description.lower(),
    )
    account = await session.scalar(stmt)
    if not account:
        return "Tabungan tidak ditemukan. Coba cek nama tabungannya."

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


async def _handle_report_request(
    session: AsyncSession, user_id: int, payload: IncomingMessage
) -> str:
    report_date = payload.timestamp.date()
    start = datetime.combine(report_date, datetime.min.time(), tzinfo=payload.timestamp.tzinfo)
    end = datetime.combine(report_date, datetime.max.time(), tzinfo=payload.timestamp.tzinfo)

    tx_stmt = select(models.Transaction).where(
        models.Transaction.user_id == user_id,
        models.Transaction.tx_datetime >= start,
        models.Transaction.tx_datetime <= end,
    )
    tx_result = await session.execute(tx_stmt)
    transactions = [row[0] for row in tx_result.all()]

    if not transactions:
        return "Belum ada transaksi hari ini."

    total_income = sum(float(t.amount) for t in transactions if t.direction == "income")
    total_expense = sum(float(t.amount) for t in transactions if t.direction == "expense")

    summary = ["Laporan harian:"]
    summary.append(f"• Pemasukan: {_format_currency(total_income, 'IDR')}")
    summary.append(f"• Pengeluaran: {_format_currency(total_expense, 'IDR')}")
    balance = total_income - total_expense
    summary.append(f"• Selisih: {_format_currency(balance, 'IDR')}")
    return "\n".join(summary)


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


def _heuristic_parse(text: str) -> ParsedIntent:
    lowered = text.lower()
    intent = "create_transaction"
    if any(word in lowered for word in ["buat tabungan", "nabung", "tabungan"]):
        intent = "create_saving"
    if any(word in lowered for word in ["setor", "deposit"]):
        intent = "deposit_saving"
    if any(word in lowered for word in ["tarik", "withdraw"]):
        intent = "withdraw_saving"
    if "laporan" in lowered:
        intent = "get_report"

    amount = _extract_amount_indonesia(text)
    if amount is None:
        amount_match = re.search(r"([0-9]+[.,0-9]*)", text)
        amount = float(amount_match.group(1).replace(".", "").replace(",", ".")) if amount_match else None

    category = _suggest_category(lowered)
    direction = _heuristic_direction(lowered)
    if direction is None and category in _INCOME_CATEGORIES:
        direction = "income"

    description = text.strip()
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


def _detect_direction(text: str) -> str:
    lowered = text.lower()
    heur = _heuristic_direction(lowered)
    if heur:
        return heur
    return "expense"


def _format_currency(amount: float, currency: str) -> str:
    prefix = "Rp" if currency.upper() == "IDR" else f"{currency.upper()} "
    return f"{prefix}{float(amount):,.0f}"


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:  # pragma: no cover - invalid formats fallback
            return None
    return None
