# Perbaikan Fitur OCR - Keuangan Bot

## Ringkasan Masalah Sebelumnya

### Masalah yang Ditemukan pada Implementasi Lama:

| No | Masalah | Dampak |
|----|---------|--------|
| 1 | **Tidak ada image preprocessing** | Gambar blur/gelap gagal di-OCR |
| 2 | **Pattern total terlalu sederhana** | Hanya cari kata "total", miss banyak format |
| 3 | **Format tanggal Indonesia tidak di-support** | "25 Des 2024" tidak terbaca |
| 4 | **Merchant detection naif** | Hanya ambil baris pertama |
| 5 | **Tidak handle format struk Indonesia** | Indomaret, Alfamart punya format berbeda |
| 6 | **Tidak ada confidence scoring** | Tidak tahu seberapa akurat hasil |
| 7 | **Tidak handle rotasi gambar** | Foto miring gagal |
| 8 | **Amount parsing salah** | "50.000" dibaca sebagai 50.0 bukan 50000 |

---

## Solusi yang Diimplementasi

### 1. Enhanced OCR Processor (`ocr_processor.py`)

File baru dengan fitur:

#### A. Image Preprocessing Pipeline
```python
# Urutan preprocessing:
1. Convert to RGB (handle RGBA/P mode)
2. EXIF transpose (auto-rotate based on camera orientation)
3. Resize (upscale jika < 500px, downscale jika > 3000px)
4. Grayscale conversion
5. Contrast enhancement (1.5x)
6. Sharpness enhancement (2.0x)
7. Median filter (denoise)
8. Binarization (threshold 140)
```

#### B. Smart Total Extraction
Multiple patterns dengan prioritas:
```python
TOTAL_PATTERNS = [
    # Label eksplisit (prioritas tinggi)
    r"(?:grand\s*)?total\s*(?:bayar|belanja|pembelian)...",
    r"(?:jumlah|jml)\s*(?:bayar|total)...",
    r"(?:bayar|tunai|cash|debit|kredit|qris|ovo|gopay)...",
    
    # Format Rp di depan
    r"[Rr][Pp]\.?\s*([0-9][0-9.,]{3,})\s*$",
    
    # Fallback: angka besar di akhir
    r"^\s*([0-9][0-9.,]{4,})\s*$",
]
```

#### C. Indonesian Date Format Support
```python
# Format yang di-support:
- 25/12/2024, 25-12-2024 (DMY)
- 2024-12-25, 2024/12/25 (YMD)
- 25/12/24 (short year)
- 25 Des 2024, 25 Desember 2024 (Indonesian month)
- 25 Dec 2024, 25 December 2024 (English month)
```

#### D. Merchant Detection Database
```python
MERCHANT_KEYWORDS = {
    INDOMARET: ["indomaret", "i-saku", "pt indomarco"],
    ALFAMART: ["alfamart", "alfa", "pt sumber alfaria"],
    SUPERMARKET: ["hypermart", "carrefour", "giant", "superindo"],
    RESTAURANT: ["resto", "mcd", "kfc", "pizza hut", "hokben"],
    CAFE: ["starbucks", "kopi kenangan", "janji jiwa"],
    ONLINE_SHOP: ["tokopedia", "shopee", "lazada"],
}
```

#### E. Amount Parsing Indonesia
```python
# Handle berbagai format:
- "1.234.567" → 1234567 (titik = ribuan)
- "1.234,56" → 1234.56 (koma = desimal)
- "1,234.56" → 1234.56 (English format)
- "50000" → 50000 (plain number)
```

#### F. Payment Method Detection
```python
PAYMENT_PATTERNS = [
    ("tunai|cash", "Tunai"),
    ("debit|kartu debit", "Debit"),
    ("qris", "QRIS"),
    ("gopay", "GoPay"),
    ("ovo", "OVO"),
    ("dana", "DANA"),
    ("shopeepay", "ShopeePay"),
]
```

#### G. Confidence Scoring
```python
# Scoring berdasarkan data yang berhasil di-extract:
- Total found: +35 points
- Merchant found: +15 points
- Merchant type recognized: +10 points (bonus)
- Date found: +15 points
- Time found: +5 points
- Items found: +2 per item (max 10)
- Payment method: +5 points
- Raw text quality: +5 points

# Final score = points / 100 (0.0 - 1.0)
```

---

### 2. Enhanced OCR Response

Response sekarang lebih lengkap:

```json
{
  "total": 132500.0,
  "subtotal": 125000.0,
  "tax": 7500.0,
  "discount": null,
  "date": "2024-12-25",
  "time": "14:30:25",
  "merchant": "Indomaret",
  "merchant_address": null,
  "receipt_type": "indomaret",
  "items": [
    {"name": "AQUA 600ML", "quantity": 1, "price": 3500, "subtotal": 3500},
    {"name": "INDOMIE GORENG", "quantity": 2, "price": 3000, "subtotal": 6000}
  ],
  "payment_method": "Tunai",
  "raw_text": "...",
  "confidence": 0.85,
  "preprocessing_applied": ["grayscale", "contrast_1.5", "binarize_140"]
}
```

---

### 3. Backend Integration

Update `_format_ocr_response()` di `wa.py`:

```python
def _format_ocr_response(ocr_response: dict) -> str:
    """Format OCR response untuk NLU parsing"""
    parts = []
    
    # Structured data
    if merchant: parts.append(f"struk dari {merchant}")
    if total: parts.append(f"total {total}")
    if date: parts.append(f"tanggal {date}")
    if payment: parts.append(f"bayar {payment}")
    if items: parts.append(f"item: {item_names}")
    
    # Low confidence warning
    if confidence < 0.5:
        logger.warning("Low OCR confidence", ...)
    
    return " ".join(parts)
```

---

## Perbandingan Hasil

### Contoh 1: Struk Indomaret

**Input Image:** Foto struk Indomaret

**Hasil Lama:**
```json
{
  "total": null,  // GAGAL - pattern tidak match
  "date": null,   // GAGAL - format "25/12/24" tidak di-support
  "merchant": "INDOMARET",
  "raw_text": "INDOMARET\n25/12/24 14:30\n..."
}
```

**Hasil Baru:**
```json
{
  "total": 132500.0,  // ✅ Berhasil
  "date": "2024-12-25",  // ✅ Berhasil
  "time": "14:30:00",  // ✅ Baru
  "merchant": "Indomaret",
  "receipt_type": "indomaret",  // ✅ Baru
  "payment_method": "Tunai",  // ✅ Baru
  "confidence": 0.85  // ✅ Baru
}
```

### Contoh 2: Struk Cafe

**Input:** Foto struk Starbucks

**Hasil Lama:**
```json
{
  "total": 65.0,  // SALAH - "65.000" dibaca sebagai 65
  "date": null,
  "merchant": "STARBUCKS COFFEE"
}
```

**Hasil Baru:**
```json
{
  "total": 110000.0,  // ✅ Grand total benar
  "subtotal": 100000.0,  // ✅ Subtotal
  "tax": 10000.0,  // ✅ PB1 10%
  "date": "2024-12-06",
  "merchant": "Starbucks",
  "receipt_type": "cafe",
  "payment_method": "GoPay",
  "confidence": 0.82
}
```

---

## File yang Diubah/Ditambah

| File | Aksi | Deskripsi |
|------|------|-----------|
| `services/ai-media-service/app/ocr_processor.py` | **NEW** | Enhanced OCR processor |
| `services/ai-media-service/app/routes.py` | **MODIFIED** | Integrate new processor |
| `services/ai-media-service/tests/test_ocr_processor.py` | **NEW** | Unit tests |
| `services/backend-api/app/services/wa.py` | **MODIFIED** | Better OCR response handling |

---

## Testing

### Run Unit Tests
```bash
cd services/ai-media-service
pytest tests/test_ocr_processor.py -v
```

### Manual Testing
```bash
# Start services
docker compose up -d

# Test OCR endpoint
curl -X POST http://localhost:8500/media/ocr \
  -H "Content-Type: application/json" \
  -d '{"media_url": "http://minio:9000/wa-media/test-receipt.jpg"}'
```

---

## Rekomendasi Lanjutan

### Short-term
1. **Add more merchant keywords** - Expand database untuk merchant lokal
2. **Improve item extraction** - Better regex untuk berbagai format item
3. **Add image rotation detection** - Detect dan fix rotasi otomatis

### Medium-term
1. **Use PaddleOCR** - Lebih akurat untuk teks Indonesia
2. **Add receipt template matching** - Template khusus per merchant
3. **Implement async processing** - Celery worker untuk heavy OCR

### Long-term
1. **Train custom model** - Fine-tune untuk struk Indonesia
2. **Add receipt validation** - Cross-check total dengan sum of items
3. **Implement feedback loop** - User correction untuk improve model

---

## Troubleshooting

### OCR Confidence Rendah
- Pastikan gambar tidak blur
- Pastikan pencahayaan cukup
- Crop gambar hanya bagian struk

### Total Tidak Terbaca
- Check raw_text untuk lihat hasil OCR mentah
- Mungkin format baru yang belum di-support
- Tambahkan pattern baru di `TOTAL_PATTERNS`

### Tanggal Salah
- Check format tanggal di struk
- Tambahkan pattern baru di `DATE_PATTERNS`
- Perhatikan urutan day/month (Indonesia vs US format)
