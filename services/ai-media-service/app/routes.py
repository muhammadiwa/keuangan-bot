from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional, TYPE_CHECKING

import httpx
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from app.config import get_settings
from app.providers.manager import AIProviderManager
from app.providers.base import ProviderError
from app.providers.stt import STTProviderManager, STTProviderError

try:  # pragma: no cover
    from PIL import Image
    import pytesseract
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    pytesseract = None  # type: ignore


router = APIRouter()
settings = get_settings()
_ai_provider_manager: Optional[AIProviderManager] = None
_stt_provider_manager: Optional[STTProviderManager] = None


def get_ai_provider_manager() -> AIProviderManager:
    """Get or create the AI Provider Manager singleton."""
    global _ai_provider_manager
    if _ai_provider_manager is None:
        _ai_provider_manager = AIProviderManager(settings)
    return _ai_provider_manager


def get_stt_provider_manager() -> STTProviderManager:
    """Get or create the STT Provider Manager singleton."""
    global _stt_provider_manager
    if _stt_provider_manager is None:
        _stt_provider_manager = STTProviderManager(settings)
    return _stt_provider_manager


class ParseRequest(BaseModel):
    text: str


class ParseResponse(BaseModel):
    intent: str
    amount: float | None = None
    currency: str | None = None
    description: str | None = None
    category_suggestion: str | None = None
    datetime: str | None = None
    confidence: float | None = None
    direction: str | None = None


class MediaUrlRequest(BaseModel):
    media_url: str


class STTResponse(BaseModel):
    text: str


class OCRResponse(BaseModel):
    total: float | None = None
    subtotal: float | None = None
    tax: float | None = None
    discount: float | None = None
    date: str | None = None
    time: str | None = None
    merchant: str | None = None
    merchant_address: str | None = None
    receipt_type: str | None = None
    items: list[dict] | None = None
    payment_method: str | None = None
    raw_text: str | None = None
    confidence: float | None = None
    preprocessing_applied: list[str] | None = None


@router.post("/ai/parse", response_model=ParseResponse)
async def parse_text(body: ParseRequest) -> ParseResponse:
    """
    Parse text using AI provider with automatic fallback.
    
    Uses the configured AI provider (via AIProviderManager) to parse
    natural language input. Falls back to heuristic parsing if all
    AI providers fail.
    """
    manager = get_ai_provider_manager()
    messages = [
        {"role": "system", "content": _get_nlu_system_prompt()},
        {"role": "user", "content": body.text},
    ]
    
    logger.debug(
        "Sending NLU request",
        provider=settings.ai_provider,
        text_length=len(body.text),
    )
    
    try:
        # Try AI providers with heuristic fallback
        ai_response, heuristic_result = await manager.chat_completion_with_heuristic_fallback(
            messages=messages,
            heuristic_fn=_heuristic_payload,
            temperature=0.3,  # Lower temperature for more consistent parsing
            max_tokens=500,
        )
        
        if ai_response:
            # AI provider succeeded
            payload = _extract_json_payload(ai_response.content)
            logger.info(
                "AI parse succeeded",
                provider=ai_response.provider,
                model=ai_response.model,
                latency_ms=ai_response.latency_ms,
            )
        else:
            # Heuristic fallback was used
            payload = heuristic_result
            logger.info("Using heuristic fallback result")
            
    except ProviderError as exc:
        # All providers and heuristic failed - this shouldn't happen
        # but handle gracefully
        logger.error("All parsing methods failed", error=str(exc))
        payload = _heuristic_payload(body.text)
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.warning("Unexpected error in parse, using heuristics", error=str(exc))
        payload = _heuristic_payload(body.text)

    return ParseResponse(**payload)


@router.post("/media/stt", response_model=STTResponse)
async def media_stt(body: MediaUrlRequest) -> STTResponse:
    """
    Transcribe audio to text using the configured STT provider.
    
    Supports multiple STT providers:
    - Local Whisper (faster-whisper)
    - OpenAI Whisper API
    - Groq Whisper API (fast transcription)
    
    Configure via STT_PROVIDER environment variable.
    """
    if not body.media_url:
        raise HTTPException(status_code=400, detail="media_url is required")
    
    audio_bytes = await _download_media(body.media_url)
    
    stt_manager = get_stt_provider_manager()
    
    try:
        result = await stt_manager.transcribe(
            audio_bytes=audio_bytes,
            language="id",  # Default to Indonesian
        )
        
        logger.info(
            "STT completed",
            provider=result.provider,
            model=result.model,
            latency_ms=round(result.latency_ms, 2),
            text_length=len(result.text),
        )
        
        return STTResponse(text=result.text)
        
    except STTProviderError as e:
        logger.error("STT provider error", error=str(e), provider=e.provider)
        raise HTTPException(
            status_code=503,
            detail=f"STT service unavailable: {e}",
        )


@router.post("/media/ocr", response_model=OCRResponse)
async def media_ocr(body: MediaUrlRequest) -> OCRResponse:
    if not body.media_url:
        raise HTTPException(status_code=400, detail="media_url is required")
    image_bytes = await _download_media(body.media_url)
    
    # Use enhanced OCR processor
    from app.ocr_processor import get_ocr_processor
    
    loop = asyncio.get_running_loop()
    
    def _process_ocr() -> dict:
        processor = get_ocr_processor(settings.tesseract_lang)
        result = processor.process_image(image_bytes)
        return result.to_dict()
    
    try:
        result = await loop.run_in_executor(None, _process_ocr)
        logger.info(
            "OCR completed",
            merchant=result.get("merchant"),
            total=result.get("total"),
            confidence=result.get("confidence"),
        )
        return OCRResponse(**result)
    except Exception as e:
        logger.exception(f"OCR processing failed: {e}")
        # Fallback to legacy OCR
        result = await _run_ocr_legacy(image_bytes)
        return OCRResponse(**result)


@router.get("/healthz", response_model=dict)
async def healthz() -> dict:
    """
    Health check endpoint with AI and STT provider status.
    
    Returns the overall service health including:
    - AI provider health status (primary and fallback)
    - STT provider health status
    - Provider latency metrics
    - Configuration details
    """
    ai_manager = get_ai_provider_manager()
    stt_manager = get_stt_provider_manager()
    
    # Get AI provider health status
    try:
        ai_health = await ai_manager.get_health_status()
    except Exception as e:
        logger.warning("Failed to get AI provider health", error=str(e))
        ai_health = {
            "status": "unknown",
            "error": str(e),
            "providers": {},
        }
    
    # Get STT provider health status
    try:
        stt_health = await stt_manager.get_health_status()
    except Exception as e:
        logger.warning("Failed to get STT provider health", error=str(e))
        stt_health = {
            "status": "unknown",
            "error": str(e),
        }
    
    # Determine overall service status
    ai_status = ai_health.get("status", "unknown")
    stt_status = stt_health.get("status", "unknown")
    
    if ai_status == "healthy" and stt_status == "healthy":
        overall_status = "ok"
    elif ai_status in ("healthy", "degraded") or stt_status in ("healthy", "degraded"):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    # Get provider info
    provider_info = ai_manager.get_provider_info()
    stt_provider_info = stt_manager.get_provider_info()
    
    return {
        "status": overall_status,
        "ai_providers": {
            "status": ai_status,
            "primary": {
                "provider": settings.ai_provider,
                "model": settings.get_effective_ai_model(),
                "available": provider_info.get("primary", {}).get("available", False),
                "health": ai_health.get("providers", {}).get("primary", {}),
            },
            "fallback": {
                "provider": settings.ai_fallback_provider,
                "model": settings.ai_fallback_model,
                "available": provider_info.get("fallback", {}).get("available", False),
                "health": ai_health.get("providers", {}).get("fallback", {}),
            } if settings.ai_fallback_provider else None,
            "summary": ai_health.get("summary", {}),
        },
        "stt_provider": {
            "status": stt_status,
            "provider": stt_provider_info.get("provider"),
            "model": stt_provider_info.get("model"),
            "health": stt_health.get("details", {}),
        },
        # Legacy fields for backward compatibility
        "ollama_url": settings.ollama_url,
        "ollama_model": settings.ollama_model,
        "whisper_model": settings.whisper_model,
        "tesseract_lang": settings.tesseract_lang,
    }


async def _call_ollama(prompt: str) -> str:
    """
    Legacy direct Ollama call (kept for backward compatibility).
    
    Note: The main /ai/parse endpoint now uses AIProviderManager.
    This function is kept for any code that still needs direct Ollama access.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


def _get_nlu_system_prompt() -> str:
    """Get the system prompt for NLU parsing."""
    return (
        "Kamu adalah agen NLU untuk pencatat keuangan pribadi. "
        "Analisis pesan pengguna dan hasilkan JSON dengan format:\n"
        "{\n"
        '  "intent": "create_transaction | create_saving | deposit_saving | withdraw_saving | get_report | set_budget | smalltalk",\n'
        '  "amount": <angka atau null>,\n'
        '  "currency": "IDR" atau kode lain,\n'
        '  "description": "ringkasan singkat",\n'
        '  "category_suggestion": "kategori",\n'
        '  "datetime": "ISO8601 atau null",\n'
        '  "confidence": 0-1\n'
        "}\n"
        "Balas hanya dengan JSON valid, tanpa penjelasan tambahan."
    )


def _build_nlu_prompt(text: str) -> str:
    """Build NLU prompt for legacy Ollama generate endpoint (kept for backward compatibility)."""
    return (
        _get_nlu_system_prompt()
        + "\n\nPesan: "
        + text
    )


def _extract_json_payload(response_text: str) -> dict[str, Any]:
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        raise ValueError("JSON payload not found in Ollama response")
    payload = json.loads(json_match.group(0))
    payload.setdefault("intent", "smalltalk")
    return payload


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
        "kuliner",
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
    "invoice",
    "customer",
    "klien",
    "transfer masuk",
    "omzet",
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


def _heuristic_payload(text: str) -> dict[str, Any]:
    lowered = text.lower()
    intent = "create_transaction"
    transactions_keywords = [
        "daftar transaksi",
        "lihat transaksi",
        "cek transaksi",
        "riwayat transaksi",
        "history transaksi",
    ]
    savings_keywords = [
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
    elif any(keyword in lowered for keyword in savings_keywords):
        intent = "list_savings"
    elif any(keyword in lowered for keyword in category_list_keywords):
        intent = "list_categories"
    elif any(keyword in lowered for keyword in category_add_keywords):
        intent = "add_category"
    elif any(keyword in lowered for keyword in category_delete_keywords):
        intent = "delete_category"
    elif any(keyword in lowered for keyword in category_rename_keywords):
        intent = "rename_category"
    elif "tabungan" in lowered and any(word in lowered for word in ["buat", "bikin", "baru"]):
        intent = "create_saving"
    elif "setor" in lowered:
        intent = "deposit_saving"
    elif "tarik" in lowered:
        intent = "withdraw_saving"
    elif "laporan" in lowered:
        intent = "get_report"

    amount = _extract_amount_indonesia(text)
    if amount is None:
        amount_match = re.search(r"([0-9]+[.,0-9]*)", text)
        if amount_match:
            amount = float(amount_match.group(1).replace(".", "").replace(",", "."))

    category = _suggest_category(lowered)
    direction = _heuristic_direction(lowered)

    if direction is None and category in _INCOME_CATEGORIES:
        direction = "income"

    return {
        "intent": intent,
        "amount": amount,
        "currency": "IDR",
        "description": text.strip(),
        "category_suggestion": category,
        "direction": direction,
        "datetime": datetime.utcnow().isoformat(),
        "confidence": 0.4,
    }


async def _download_media(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def _transcribe_audio(audio_bytes: bytes) -> str:
    if WhisperModel is None:  # pragma: no cover
        logger.warning("faster-whisper not installed, returning placeholder transcript")
        return "(transkripsi belum tersedia)"

    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")

    with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    loop = asyncio.get_running_loop()

    def _run_transcribe() -> str:
        segments, _ = _whisper_model.transcribe(str(tmp_path))
        text_parts = [segment.text.strip() for segment in segments]
        return " ".join(part for part in text_parts if part)

    try:
        transcript = await loop.run_in_executor(None, _run_transcribe)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:  # pragma: no cover
            logger.warning("Failed to remove temp audio", path=str(tmp_path))

    return transcript or "(tidak ada transkrip)"


async def _run_ocr_legacy(image_bytes: bytes) -> dict[str, Any]:
    """Legacy OCR fallback jika enhanced processor gagal"""
    if pytesseract is None or Image is None:  # pragma: no cover
        logger.warning("OCR dependencies missing, returning empty result")
        return {"total": None, "date": None, "merchant": None, "raw_text": None}

    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)

    loop = asyncio.get_running_loop()

    def _run() -> dict[str, Any]:
        image = Image.open(tmp_path)
        text = pytesseract.image_to_string(image, lang=settings.tesseract_lang)
        merchant = _extract_merchant_legacy(text)
        total = _extract_total_legacy(text)
        ocr_date = _extract_date_legacy(text)
        return {
            "total": total,
            "date": ocr_date,
            "merchant": merchant,
            "raw_text": text,
            "confidence": 0.3,  # Low confidence for legacy
        }

    try:
        result = await loop.run_in_executor(None, _run)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:  # pragma: no cover
            logger.warning("Failed to remove temp image", path=str(tmp_path))

    return result


def _extract_total_legacy(text: str) -> float | None:
    """Legacy total extraction"""
    # Try explicit total patterns first
    patterns = [
        r"(?:grand\s*)?total\s*[:\s]*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
        r"(?:bayar|tunai|cash)\s*[:\s]*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(".", "").replace(",", ".")
            try:
                amount = float(value)
                if amount > 100:
                    return amount
            except ValueError:
                continue
    
    # Fallback: find largest number
    matches = re.findall(r"([0-9][0-9.,]{4,})", text)
    amounts = []
    for m in matches:
        try:
            val = float(m.replace(".", "").replace(",", "."))
            if val > 100:
                amounts.append(val)
        except ValueError:
            continue
    
    return max(amounts) if amounts else None


def _extract_date_legacy(text: str) -> str | None:
    """Legacy date extraction dengan support format Indonesia"""
    patterns = [
        (r"(\d{4})-(\d{2})-(\d{2})", "ymd"),
        (r"(\d{2})/(\d{2})/(\d{4})", "dmy"),
        (r"(\d{2})-(\d{2})-(\d{4})", "dmy"),
        (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", "dmy"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                groups = match.groups()
                if fmt == "ymd":
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                else:
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    if year < 100:
                        year += 2000
                
                if 1 <= day <= 31 and 1 <= month <= 12 and 2000 <= year <= 2100:
                    return f"{year:04d}-{month:02d}-{day:02d}"
            except (ValueError, IndexError):
                continue
    return None


def _extract_merchant_legacy(text: str) -> str | None:
    """Legacy merchant extraction dengan keyword detection"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    
    # Known merchants
    known_merchants = [
        "indomaret", "alfamart", "alfamidi", "hypermart", "carrefour",
        "giant", "superindo", "mcd", "mcdonald", "kfc", "starbucks",
        "kopi kenangan", "janji jiwa"
    ]
    
    text_lower = text.lower()
    for merchant in known_merchants:
        if merchant in text_lower:
            return merchant.title()
    
    # Fallback: first non-numeric line
    for line in lines[:3]:
        if not re.match(r'^[\d\s/\-:.,]+$', line) and len(line) > 2:
            return line[:64]
    
    return lines[0][:64]
