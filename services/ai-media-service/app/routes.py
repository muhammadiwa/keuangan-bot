import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from app.config import get_settings

try:  # pragma: no cover - optional dependencies for heavy workloads
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None  # type: ignore

try:  # pragma: no cover
    from PIL import Image
    import pytesseract
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    pytesseract = None  # type: ignore


router = APIRouter()
settings = get_settings()
_whisper_model: WhisperModel | None = None


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
    date: str | None = None
    merchant: str | None = None
    raw_text: str | None = None


@router.post("/ai/parse", response_model=ParseResponse)
async def parse_text(body: ParseRequest) -> ParseResponse:
    prompt = _build_nlu_prompt(body.text)
    logger.debug("Sending prompt to Ollama", prompt=prompt)
    try:
        result = await _call_ollama(prompt)
        payload = _extract_json_payload(result)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("Ollama parse failed, using heuristics", error=str(exc))
        payload = _heuristic_payload(body.text)

    return ParseResponse(**payload)


@router.post("/media/stt", response_model=STTResponse)
async def media_stt(body: MediaUrlRequest) -> STTResponse:
    if not body.media_url:
        raise HTTPException(status_code=400, detail="media_url is required")
    audio_bytes = await _download_media(body.media_url)
    text = await _transcribe_audio(audio_bytes)
    return STTResponse(text=text)


@router.post("/media/ocr", response_model=OCRResponse)
async def media_ocr(body: MediaUrlRequest) -> OCRResponse:
    if not body.media_url:
        raise HTTPException(status_code=400, detail="media_url is required")
    image_bytes = await _download_media(body.media_url)
    result = await _run_ocr(image_bytes)
    return OCRResponse(**result)


@router.get("/healthz", response_model=dict)
async def healthz() -> dict:
    return {
        "status": "ok",
        "ollama_url": settings.ollama_url,
        "ollama_model": settings.ollama_model,
        "whisper_model": settings.whisper_model,
        "tesseract_lang": settings.tesseract_lang,
    }


async def _call_ollama(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


def _build_nlu_prompt(text: str) -> str:
    return (
        "Kamu adalah agen NLU untuk pencatat keuangan pribadi. "
        "Analisis pesan berikut dan hasilkan JSON dengan format:\n"
        "{\n"
        '  "intent": "create_transaction | create_saving | deposit_saving | withdraw_saving | get_report | set_budget | smalltalk",\n'
        '  "amount": <angka atau null>,\n'
        '  "currency": "IDR" atau kode lain,\n'
        '  "description": "ringkasan singkat",\n'
        '  "category_suggestion": "kategori",\n'
        '  "datetime": "ISO8601 atau null",\n'
        '  "confidence": 0-1\n'
        "}\n"
        "Pesan: "
        f"""{text}"""
        "\nBalas hanya JSON valid."
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
    if "tabungan" in lowered and any(word in lowered for word in ["buat", "bikin", "baru"]):
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


async def _run_ocr(image_bytes: bytes) -> dict[str, Any]:
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
        merchant = _extract_merchant(text)
        total = _extract_total(text)
        ocr_date = _extract_date(text)
        return {
            "total": total,
            "date": ocr_date,
            "merchant": merchant,
            "raw_text": text,
        }

    try:
        result = await loop.run_in_executor(None, _run)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:  # pragma: no cover
            logger.warning("Failed to remove temp image", path=str(tmp_path))

    return result


def _extract_total(text: str) -> float | None:
    pattern = re.compile(r"total\s*[:=]?\s*([0-9.,]+)", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        match = re.search(r"([0-9][0-9.,]{4,})", text)
    if not match:
        return None
    value = match.group(1).replace(".", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:  # pragma: no cover
        return None


def _extract_date(text: str) -> str | None:
    patterns = [
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{2}/\d{2}/\d{4})",
        r"(\d{2}-\d{2}-\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            raw = match.group(1)
            try:
                dt = datetime.fromisoformat(raw.replace("/", "-") if "/" in raw else raw)
                return dt.date().isoformat()
            except ValueError:
                continue
    return None


def _extract_merchant(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[0][:64]
    return None
