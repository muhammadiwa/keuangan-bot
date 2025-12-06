"""
Enhanced OCR Processor untuk Struk Indonesia

Fitur:
- Image preprocessing (grayscale, contrast, denoise, deskew)
- Multiple receipt format support (Indomaret, Alfamart, minimarket, restaurant)
- Indonesian date format parsing
- Smart total extraction dengan multiple patterns
- Merchant detection dari database keywords
- Confidence scoring
- Item extraction dari struk
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

from loguru import logger

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None
    ImageEnhance = None
    ImageFilter = None
    ImageOps = None
    np = None

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    pytesseract = None


class ReceiptType(Enum):
    """Jenis struk yang dikenali"""
    INDOMARET = "indomaret"
    ALFAMART = "alfamart"
    ALFAMIDI = "alfamidi"
    SUPERMARKET = "supermarket"
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    ONLINE_SHOP = "online_shop"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class ReceiptItem:
    """Item dalam struk"""
    name: str
    quantity: int = 1
    price: float = 0.0
    subtotal: float = 0.0


@dataclass
class OCRResult:
    """Hasil OCR yang terstruktur"""
    total: float | None = None
    subtotal: float | None = None
    tax: float | None = None
    discount: float | None = None
    date: str | None = None
    time: str | None = None
    merchant: str | None = None
    merchant_address: str | None = None
    receipt_type: ReceiptType = ReceiptType.UNKNOWN
    items: list[ReceiptItem] = field(default_factory=list)
    payment_method: str | None = None
    raw_text: str = ""
    confidence: float = 0.0
    preprocessing_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "subtotal": self.subtotal,
            "tax": self.tax,
            "discount": self.discount,
            "date": self.date,
            "time": self.time,
            "merchant": self.merchant,
            "merchant_address": self.merchant_address,
            "receipt_type": self.receipt_type.value,
            "items": [
                {"name": i.name, "quantity": i.quantity, "price": i.price, "subtotal": i.subtotal}
                for i in self.items
            ],
            "payment_method": self.payment_method,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "preprocessing_applied": self.preprocessing_applied,
        }


# Database merchant keywords untuk deteksi
MERCHANT_KEYWORDS = {
    ReceiptType.INDOMARET: [
        "indomaret", "i-saku", "isaku", "pt indomarco", "indomarco prismatama"
    ],
    ReceiptType.ALFAMART: [
        "alfamart", "alfa", "pt sumber alfaria", "alfaria trijaya"
    ],
    ReceiptType.ALFAMIDI: [
        "alfamidi", "midi utama"
    ],
    ReceiptType.SUPERMARKET: [
        "hypermart", "carrefour", "transmart", "giant", "superindo", 
        "hero", "lotte mart", "farmers market", "ranch market",
        "foodhall", "grand lucky", "hari hari", "tip top"
    ],
    ReceiptType.RESTAURANT: [
        "resto", "restaurant", "restoran", "warung", "rumah makan",
        "cafe", "kafe", "coffee", "kopi", "mcd", "mcdonald", 
        "kfc", "burger king", "pizza hut", "domino", "hokben",
        "yoshinoya", "marugame", "solaria", "es teler", "bakmi gm"
    ],
    ReceiptType.CAFE: [
        "starbucks", "sbux", "kopi kenangan", "janji jiwa", "fore coffee",
        "tomoro", "point coffee", "excelso", "anomali", "djournal"
    ],
    ReceiptType.ONLINE_SHOP: [
        "tokopedia", "shopee", "lazada", "bukalapak", "blibli",
        "jd.id", "zalora", "sociolla"
    ],
}

# Pattern untuk ekstraksi total - urut dari paling spesifik
TOTAL_PATTERNS = [
    # Format dengan label eksplisit
    r"(?:grand\s*)?total\s*(?:bayar|belanja|pembelian|harga|:|\s)\s*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
    r"(?:jumlah|jml)\s*(?:bayar|total|:|\s)\s*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
    r"(?:total|ttl)\s*[:\s]*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
    r"(?:bayar|tunai|cash|debit|kredit|qris|ovo|gopay|dana|shopeepay)\s*[:\s]*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
    # Format Rp di depan
    r"[Rr][Pp]\.?\s*([0-9][0-9.,]{3,})\s*$",
    # Angka besar di akhir baris (likely total)
    r"^\s*([0-9][0-9.,]{4,})\s*$",
]

# Pattern untuk subtotal
SUBTOTAL_PATTERNS = [
    r"(?:sub\s*total|subtotal)\s*[:\s]*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
    r"(?:total\s*item|total\s*belanja)\s*[:\s]*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
]

# Pattern untuk pajak
TAX_PATTERNS = [
    r"(?:ppn|pajak|tax|vat)\s*(?:\d+%?)?\s*[:\s]*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
    r"(?:pb1|service)\s*(?:\d+%?)?\s*[:\s]*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
]

# Pattern untuk diskon
DISCOUNT_PATTERNS = [
    r"(?:diskon|discount|potongan|promo|hemat)\s*[:\s]*-?\s*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
    r"(?:voucher|kupon|coupon)\s*[:\s]*-?\s*[Rr]?[Pp]?\.?\s*([0-9][0-9.,]*)",
]

# Pattern untuk tanggal Indonesia
DATE_PATTERNS = [
    # Format: 25/12/2024, 25-12-2024
    (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "dmy"),
    (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})", "dmy_short"),
    # Format: 2024-12-25, 2024/12/25
    (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
    # Format: 25 Des 2024, 25 Desember 2024
    (r"(\d{1,2})\s*(jan|feb|mar|apr|mei|jun|jul|agu|sep|okt|nov|des)[a-z]*\s*(\d{4})", "dmy_text"),
    (r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})", "dmy_text_en"),
]

# Mapping bulan Indonesia
MONTH_MAP_ID = {
    "jan": 1, "januari": 1,
    "feb": 2, "februari": 2,
    "mar": 3, "maret": 3,
    "apr": 4, "april": 4,
    "mei": 5,
    "jun": 6, "juni": 6,
    "jul": 7, "juli": 7,
    "agu": 8, "agustus": 8,
    "sep": 9, "september": 9,
    "okt": 10, "oktober": 10,
    "nov": 11, "november": 11,
    "des": 12, "desember": 12,
}

MONTH_MAP_EN = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Pattern untuk waktu
TIME_PATTERNS = [
    r"(\d{1,2})[:\.](\d{2})(?:[:\.](\d{2}))?\s*(?:wib|wit|wita)?",
    r"(?:jam|pukul|time)\s*[:\s]*(\d{1,2})[:\.](\d{2})",
]

# Pattern untuk metode pembayaran
PAYMENT_PATTERNS = [
    (r"(?:tunai|cash)", "Tunai"),
    (r"(?:debit|kartu debit)", "Debit"),
    (r"(?:kredit|kartu kredit|cc)", "Kredit"),
    (r"(?:qris)", "QRIS"),
    (r"(?:gopay|go-pay)", "GoPay"),
    (r"(?:ovo)", "OVO"),
    (r"(?:dana)", "DANA"),
    (r"(?:shopeepay|spay)", "ShopeePay"),
    (r"(?:linkaja|link aja)", "LinkAja"),
    (r"(?:i-saku|isaku)", "i-Saku"),
    (r"(?:ponta|member)", "Member/Ponta"),
]


class OCRProcessor:
    """Enhanced OCR Processor dengan preprocessing dan smart extraction"""

    def __init__(self, tesseract_lang: str = "eng+ind"):
        self.tesseract_lang = tesseract_lang
        self.preprocessing_steps: list[str] = []

    def process_image(self, image_bytes: bytes) -> OCRResult:
        """Main entry point untuk proses OCR"""
        if not HAS_PIL or not HAS_TESSERACT:
            logger.warning("OCR dependencies not available")
            return OCRResult(confidence=0.0)

        self.preprocessing_steps = []
        
        try:
            # Load image
            image = Image.open(BytesIO(image_bytes))
            logger.debug(f"Loaded image: {image.size}, mode: {image.mode}")

            # Preprocessing pipeline
            processed_image = self._preprocess_image(image)

            # Run OCR dengan multiple configs
            raw_text = self._run_tesseract(processed_image)
            logger.debug(f"OCR raw text length: {len(raw_text)}")

            # Parse hasil OCR
            result = self._parse_receipt(raw_text)
            result.preprocessing_applied = self.preprocessing_steps.copy()

            # Calculate confidence
            result.confidence = self._calculate_confidence(result)

            return result

        except Exception as e:
            logger.exception(f"OCR processing failed: {e}")
            return OCRResult(confidence=0.0, raw_text=str(e))

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Pipeline preprocessing untuk meningkatkan akurasi OCR"""
        
        # 1. Convert to RGB if needed
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
            self.preprocessing_steps.append("convert_rgb")

        # 2. Auto-rotate based on EXIF
        image = ImageOps.exif_transpose(image)
        self.preprocessing_steps.append("exif_transpose")

        # 3. Resize if too small or too large
        width, height = image.size
        if width < 500 or height < 500:
            scale = max(500 / width, 500 / height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            self.preprocessing_steps.append(f"upscale_{new_size}")
        elif width > 3000 or height > 3000:
            scale = min(3000 / width, 3000 / height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            self.preprocessing_steps.append(f"downscale_{new_size}")

        # 4. Convert to grayscale
        gray = image.convert("L")
        self.preprocessing_steps.append("grayscale")

        # 5. Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)
        self.preprocessing_steps.append("contrast_1.5")

        # 6. Enhance sharpness
        enhancer = ImageEnhance.Sharpness(gray)
        gray = enhancer.enhance(2.0)
        self.preprocessing_steps.append("sharpness_2.0")

        # 7. Denoise dengan median filter
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        self.preprocessing_steps.append("median_filter")

        # 8. Binarization (adaptive threshold simulation)
        # Pillow doesn't have adaptive threshold, so we use simple threshold
        threshold = 140
        gray = gray.point(lambda x: 255 if x > threshold else 0, mode='1')
        gray = gray.convert("L")
        self.preprocessing_steps.append(f"binarize_{threshold}")

        return gray

    def _run_tesseract(self, image: Image.Image) -> str:
        """Run Tesseract dengan optimal config"""
        # Config untuk struk/receipt
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        
        try:
            text = pytesseract.image_to_string(
                image, 
                lang=self.tesseract_lang,
                config=custom_config
            )
            return text
        except Exception as e:
            logger.warning(f"Tesseract failed with custom config: {e}")
            # Fallback to default
            return pytesseract.image_to_string(image, lang=self.tesseract_lang)

    def _parse_receipt(self, raw_text: str) -> OCRResult:
        """Parse raw OCR text menjadi structured data"""
        result = OCRResult(raw_text=raw_text)
        
        # Normalize text
        text_lower = raw_text.lower()
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # 1. Detect receipt type & merchant
        result.receipt_type = self._detect_receipt_type(text_lower)
        result.merchant = self._extract_merchant(lines, result.receipt_type)

        # 2. Extract date & time
        result.date = self._extract_date(raw_text)
        result.time = self._extract_time(raw_text)

        # 3. Extract amounts
        result.total = self._extract_total(raw_text, lines)
        result.subtotal = self._extract_amount(raw_text, SUBTOTAL_PATTERNS)
        result.tax = self._extract_amount(raw_text, TAX_PATTERNS)
        result.discount = self._extract_amount(raw_text, DISCOUNT_PATTERNS)

        # 4. Extract payment method
        result.payment_method = self._extract_payment_method(text_lower)

        # 5. Extract items (basic)
        result.items = self._extract_items(lines)

        return result

    def _detect_receipt_type(self, text_lower: str) -> ReceiptType:
        """Deteksi jenis struk berdasarkan keywords"""
        for receipt_type, keywords in MERCHANT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return receipt_type
        return ReceiptType.UNKNOWN

    def _extract_merchant(self, lines: list[str], receipt_type: ReceiptType) -> str | None:
        """Extract nama merchant dengan smart detection"""
        if not lines:
            return None

        # Untuk tipe yang dikenal, cari keyword spesifik
        if receipt_type != ReceiptType.UNKNOWN:
            for receipt_t, keywords in MERCHANT_KEYWORDS.items():
                if receipt_t == receipt_type:
                    for line in lines[:5]:  # Check first 5 lines
                        line_lower = line.lower()
                        for kw in keywords:
                            if kw in line_lower:
                                # Clean up merchant name
                                return self._clean_merchant_name(line)

        # Fallback: ambil baris pertama yang bukan tanggal/waktu/angka saja
        for line in lines[:3]:
            # Skip jika hanya angka atau tanggal
            if re.match(r'^[\d\s/\-:.,]+$', line):
                continue
            # Skip jika terlalu pendek
            if len(line) < 3:
                continue
            return self._clean_merchant_name(line)

        return lines[0][:64] if lines else None

    def _clean_merchant_name(self, name: str) -> str:
        """Bersihkan nama merchant"""
        # Remove common prefixes
        name = re.sub(r'^(pt\.?|cv\.?|ud\.?|toko|warung)\s*', '', name, flags=re.IGNORECASE)
        # Remove trailing numbers/special chars
        name = re.sub(r'[\d\-_]+$', '', name)
        # Capitalize properly
        name = name.strip().title()
        return name[:64] if name else ""

    def _extract_date(self, text: str) -> str | None:
        """Extract tanggal dengan support format Indonesia"""
        text_lower = text.lower()
        
        for pattern, format_type in DATE_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()
                    
                    if format_type == "dmy":
                        day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    elif format_type == "dmy_short":
                        day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                        year = 2000 + year if year < 100 else year
                    elif format_type == "ymd":
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    elif format_type == "dmy_text":
                        day = int(groups[0])
                        month = MONTH_MAP_ID.get(groups[1].lower()[:3], 1)
                        year = int(groups[2])
                    elif format_type == "dmy_text_en":
                        day = int(groups[0])
                        month = MONTH_MAP_EN.get(groups[1].lower()[:3], 1)
                        year = int(groups[2])
                    else:
                        continue

                    # Validate
                    if 1 <= day <= 31 and 1 <= month <= 12 and 2000 <= year <= 2100:
                        return f"{year:04d}-{month:02d}-{day:02d}"
                        
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_time(self, text: str) -> str | None:
        """Extract waktu dari struk"""
        for pattern in TIME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    hour = int(groups[0])
                    minute = int(groups[1])
                    second = int(groups[2]) if len(groups) > 2 and groups[2] else 0
                    
                    if 0 <= hour <= 23 and 0 <= minute <= 59:
                        return f"{hour:02d}:{minute:02d}:{second:02d}"
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_total(self, text: str, lines: list[str]) -> float | None:
        """Extract total dengan multiple strategies"""
        
        # Strategy 1: Cari pattern total eksplisit
        for pattern in TOTAL_PATTERNS[:4]:  # Patterns dengan label
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                amount = self._parse_amount(match.group(1))
                if amount and amount > 100:  # Minimal Rp 100
                    return amount

        # Strategy 2: Cari angka terbesar di bagian bawah struk
        bottom_lines = lines[-10:] if len(lines) > 10 else lines
        amounts = []
        for line in bottom_lines:
            for match in re.finditer(r'([0-9][0-9.,]{3,})', line):
                amount = self._parse_amount(match.group(1))
                if amount and amount > 100:
                    amounts.append(amount)

        if amounts:
            # Ambil yang terbesar (biasanya total)
            return max(amounts)

        # Strategy 3: Fallback ke pattern angka besar manapun
        for pattern in TOTAL_PATTERNS[4:]:
            for match in re.finditer(pattern, text, re.MULTILINE):
                amount = self._parse_amount(match.group(1))
                if amount and amount > 1000:
                    return amount

        return None

    def _extract_amount(self, text: str, patterns: list[str]) -> float | None:
        """Generic amount extraction"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._parse_amount(match.group(1))
        return None

    def _parse_amount(self, amount_str: str) -> float | None:
        """Parse string amount ke float"""
        if not amount_str:
            return None
        
        # Remove spaces
        amount_str = amount_str.strip()
        
        # Handle Indonesian format: 1.234.567 atau 1,234,567
        # Detect format
        if '.' in amount_str and ',' in amount_str:
            # Mixed format - assume Indonesian (. as thousand, , as decimal)
            if amount_str.rfind('.') > amount_str.rfind(','):
                # Format: 1,234.56 (English)
                amount_str = amount_str.replace(',', '')
            else:
                # Format: 1.234,56 (Indonesian)
                amount_str = amount_str.replace('.', '').replace(',', '.')
        elif '.' in amount_str:
            # Check if it's thousand separator or decimal
            parts = amount_str.split('.')
            if len(parts[-1]) == 3:
                # Likely thousand separator: 1.234.567
                amount_str = amount_str.replace('.', '')
            # else: decimal point, keep as is
        elif ',' in amount_str:
            # Check if it's thousand separator or decimal
            parts = amount_str.split(',')
            if len(parts[-1]) == 3:
                # Likely thousand separator: 1,234,567
                amount_str = amount_str.replace(',', '')
            else:
                # Decimal separator: 1234,56
                amount_str = amount_str.replace(',', '.')

        try:
            return float(amount_str)
        except ValueError:
            return None

    def _extract_payment_method(self, text_lower: str) -> str | None:
        """Extract metode pembayaran"""
        for pattern, method in PAYMENT_PATTERNS:
            if re.search(pattern, text_lower):
                return method
        return None

    def _extract_items(self, lines: list[str]) -> list[ReceiptItem]:
        """Extract item-item dari struk (basic implementation)"""
        items = []
        
        # Pattern untuk item: nama + harga
        item_pattern = re.compile(
            r'^(.+?)\s+(\d+)\s*[xX@]\s*([0-9.,]+)\s+([0-9.,]+)$|'  # qty x price = subtotal
            r'^(.+?)\s+([0-9.,]{4,})$'  # name + price
        )

        for line in lines:
            # Skip header/footer lines
            if any(skip in line.lower() for skip in [
                'total', 'subtotal', 'pajak', 'tax', 'diskon', 'tunai', 'kembalian',
                'terima kasih', 'thank you', 'struk', 'receipt', 'kasir', 'tanggal'
            ]):
                continue

            match = item_pattern.match(line)
            if match:
                groups = match.groups()
                if groups[0]:  # First pattern matched
                    items.append(ReceiptItem(
                        name=groups[0].strip(),
                        quantity=int(groups[1]),
                        price=self._parse_amount(groups[2]) or 0,
                        subtotal=self._parse_amount(groups[3]) or 0,
                    ))
                elif groups[4]:  # Second pattern matched
                    items.append(ReceiptItem(
                        name=groups[4].strip(),
                        subtotal=self._parse_amount(groups[5]) or 0,
                    ))

        return items[:20]  # Limit to 20 items

    def _calculate_confidence(self, result: OCRResult) -> float:
        """Hitung confidence score berdasarkan data yang berhasil di-extract"""
        score = 0.0
        max_score = 100.0

        # Total found (most important)
        if result.total is not None and result.total > 0:
            score += 35.0

        # Merchant found
        if result.merchant:
            score += 15.0
            # Bonus jika merchant type dikenali
            if result.receipt_type != ReceiptType.UNKNOWN:
                score += 10.0

        # Date found
        if result.date:
            score += 15.0

        # Time found
        if result.time:
            score += 5.0

        # Items found
        if result.items:
            score += min(len(result.items) * 2, 10.0)

        # Payment method found
        if result.payment_method:
            score += 5.0

        # Raw text quality (has reasonable length)
        if len(result.raw_text) > 100:
            score += 5.0

        return min(score / max_score, 1.0)


# Singleton instance
_ocr_processor: OCRProcessor | None = None


def get_ocr_processor(tesseract_lang: str = "eng+ind") -> OCRProcessor:
    """Get or create OCR processor instance"""
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = OCRProcessor(tesseract_lang)
    return _ocr_processor
