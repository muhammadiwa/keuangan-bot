"""
Unit tests untuk OCR Processor

Test cases mencakup:
- Parsing amount format Indonesia
- Ekstraksi tanggal berbagai format
- Deteksi merchant
- Ekstraksi total dari berbagai jenis struk
"""

import pytest
from app.ocr_processor import (
    OCRProcessor,
    OCRResult,
    ReceiptType,
    MONTH_MAP_ID,
)


class TestAmountParsing:
    """Test parsing nominal uang format Indonesia"""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_parse_amount_indonesian_thousand_separator(self):
        """Format: 1.234.567 (titik sebagai pemisah ribuan)"""
        assert self.processor._parse_amount("1.234.567") == 1234567.0
        assert self.processor._parse_amount("50.000") == 50000.0
        assert self.processor._parse_amount("132.500") == 132500.0

    def test_parse_amount_with_decimal(self):
        """Format: 1.234,56 (koma sebagai desimal)"""
        assert self.processor._parse_amount("1.234,56") == 1234.56
        assert self.processor._parse_amount("50.000,00") == 50000.0

    def test_parse_amount_english_format(self):
        """Format: 1,234.56 (English format)"""
        assert self.processor._parse_amount("1,234.56") == 1234.56
        assert self.processor._parse_amount("50,000.00") == 50000.0

    def test_parse_amount_plain_number(self):
        """Format: 50000 (tanpa separator)"""
        assert self.processor._parse_amount("50000") == 50000.0
        assert self.processor._parse_amount("132500") == 132500.0

    def test_parse_amount_with_spaces(self):
        """Amount dengan spasi"""
        assert self.processor._parse_amount("  50000  ") == 50000.0

    def test_parse_amount_invalid(self):
        """Invalid amount returns None"""
        assert self.processor._parse_amount("") is None
        assert self.processor._parse_amount("abc") is None


class TestDateExtraction:
    """Test ekstraksi tanggal berbagai format"""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_extract_date_dmy_slash(self):
        """Format: 25/12/2024"""
        text = "Tanggal: 25/12/2024"
        assert self.processor._extract_date(text) == "2024-12-25"

    def test_extract_date_dmy_dash(self):
        """Format: 25-12-2024"""
        text = "Tanggal: 25-12-2024"
        assert self.processor._extract_date(text) == "2024-12-25"

    def test_extract_date_ymd(self):
        """Format: 2024-12-25"""
        text = "Date: 2024-12-25"
        assert self.processor._extract_date(text) == "2024-12-25"

    def test_extract_date_short_year(self):
        """Format: 25/12/24"""
        text = "Tgl: 25/12/24"
        assert self.processor._extract_date(text) == "2024-12-25"

    def test_extract_date_indonesian_month(self):
        """Format: 25 Des 2024"""
        text = "Tanggal: 25 Des 2024"
        assert self.processor._extract_date(text) == "2024-12-25"

    def test_extract_date_indonesian_month_full(self):
        """Format: 25 Desember 2024"""
        text = "Tanggal: 25 Desember 2024"
        assert self.processor._extract_date(text) == "2024-12-25"

    def test_extract_date_english_month(self):
        """Format: 25 Dec 2024"""
        text = "Date: 25 Dec 2024"
        assert self.processor._extract_date(text) == "2024-12-25"

    def test_extract_date_not_found(self):
        """No date in text"""
        text = "Tidak ada tanggal di sini"
        assert self.processor._extract_date(text) is None


class TestTimeExtraction:
    """Test ekstraksi waktu"""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_extract_time_colon(self):
        """Format: 14:30:00"""
        text = "Waktu: 14:30:00"
        assert self.processor._extract_time(text) == "14:30:00"

    def test_extract_time_without_seconds(self):
        """Format: 14:30"""
        text = "Jam: 14:30"
        assert self.processor._extract_time(text) == "14:30:00"

    def test_extract_time_with_wib(self):
        """Format: 14:30 WIB"""
        text = "Pukul: 14:30 WIB"
        assert self.processor._extract_time(text) == "14:30:00"


class TestMerchantDetection:
    """Test deteksi merchant"""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_detect_indomaret(self):
        """Detect Indomaret"""
        text = "INDOMARET\nJl. Sudirman No. 123"
        result = self.processor._detect_receipt_type(text.lower())
        assert result == ReceiptType.INDOMARET

    def test_detect_alfamart(self):
        """Detect Alfamart"""
        text = "ALFAMART\nJl. Gatot Subroto"
        result = self.processor._detect_receipt_type(text.lower())
        assert result == ReceiptType.ALFAMART

    def test_detect_starbucks(self):
        """Detect Starbucks"""
        text = "STARBUCKS COFFEE\nGrand Indonesia"
        result = self.processor._detect_receipt_type(text.lower())
        assert result == ReceiptType.CAFE

    def test_detect_unknown(self):
        """Unknown merchant"""
        text = "TOKO RANDOM\nJl. Unknown"
        result = self.processor._detect_receipt_type(text.lower())
        assert result == ReceiptType.UNKNOWN


class TestTotalExtraction:
    """Test ekstraksi total dari struk"""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_extract_total_explicit(self):
        """Total dengan label eksplisit"""
        text = """
        Item 1    10.000
        Item 2    20.000
        TOTAL     30.000
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        total = self.processor._extract_total(text, lines)
        assert total == 30000.0

    def test_extract_total_with_rp(self):
        """Total dengan Rp"""
        text = """
        Subtotal  Rp 25.000
        PPN 10%   Rp  2.500
        Total     Rp 27.500
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        total = self.processor._extract_total(text, lines)
        assert total == 27500.0

    def test_extract_total_grand_total(self):
        """Grand Total"""
        text = """
        Subtotal      100.000
        Diskon        -10.000
        Grand Total    90.000
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        total = self.processor._extract_total(text, lines)
        assert total == 90000.0

    def test_extract_total_bayar(self):
        """Total Bayar"""
        text = """
        Total Belanja  50.000
        Bayar Tunai    50.000
        Kembalian           0
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        total = self.processor._extract_total(text, lines)
        assert total == 50000.0

    def test_extract_total_indomaret_format(self):
        """Format struk Indomaret"""
        text = """
        INDOMARET
        JL SUDIRMAN NO 123
        25/12/2024 14:30
        
        AQUA 600ML      3.500
        INDOMIE GORENG  3.000
        ROTI TAWAR      15.000
        
        TOTAL          21.500
        TUNAI          25.000
        KEMBALIAN       3.500
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        total = self.processor._extract_total(text, lines)
        assert total == 21500.0


class TestPaymentMethodExtraction:
    """Test ekstraksi metode pembayaran"""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_extract_tunai(self):
        """Pembayaran tunai"""
        assert self.processor._extract_payment_method("bayar tunai 50000") == "Tunai"
        assert self.processor._extract_payment_method("cash 50000") == "Tunai"

    def test_extract_qris(self):
        """Pembayaran QRIS"""
        assert self.processor._extract_payment_method("pembayaran qris") == "QRIS"

    def test_extract_gopay(self):
        """Pembayaran GoPay"""
        assert self.processor._extract_payment_method("gopay 50000") == "GoPay"

    def test_extract_ovo(self):
        """Pembayaran OVO"""
        assert self.processor._extract_payment_method("ovo 50000") == "OVO"

    def test_extract_debit(self):
        """Pembayaran Debit"""
        assert self.processor._extract_payment_method("kartu debit bca") == "Debit"


class TestConfidenceCalculation:
    """Test perhitungan confidence score"""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_high_confidence(self):
        """High confidence dengan data lengkap"""
        result = OCRResult(
            total=50000.0,
            merchant="Indomaret",
            receipt_type=ReceiptType.INDOMARET,
            date="2024-12-25",
            time="14:30:00",
            payment_method="Tunai",
            raw_text="x" * 200,
        )
        confidence = self.processor._calculate_confidence(result)
        assert confidence >= 0.8

    def test_medium_confidence(self):
        """Medium confidence dengan data partial"""
        result = OCRResult(
            total=50000.0,
            merchant="Unknown Store",
            raw_text="x" * 100,
        )
        confidence = self.processor._calculate_confidence(result)
        assert 0.4 <= confidence <= 0.7

    def test_low_confidence(self):
        """Low confidence dengan data minimal"""
        result = OCRResult(
            raw_text="short text",
        )
        confidence = self.processor._calculate_confidence(result)
        assert confidence < 0.3


class TestFullReceiptParsing:
    """Test parsing struk lengkap"""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_parse_indomaret_receipt(self):
        """Parse struk Indomaret"""
        raw_text = """
        INDOMARET
        PT INDOMARCO PRISMATAMA
        JL SUDIRMAN NO 123
        JAKARTA PUSAT
        
        25/12/2024 14:30:25
        
        AQUA 600ML          1 x 3.500     3.500
        INDOMIE GORENG      2 x 3.000     6.000
        ROTI TAWAR SARI     1 x 15.000   15.000
        
        SUBTOTAL                         24.500
        DISKON MEMBER                    -2.000
        
        TOTAL                            22.500
        TUNAI                            25.000
        KEMBALIAN                         2.500
        
        TERIMA KASIH
        SELAMAT BERBELANJA
        """
        result = self.processor._parse_receipt(raw_text)
        
        assert result.receipt_type == ReceiptType.INDOMARET
        assert result.merchant is not None
        assert "indomaret" in result.merchant.lower() or "indomarco" in result.merchant.lower()
        assert result.date == "2024-12-25"
        assert result.time == "14:30:25"
        assert result.total == 22500.0
        assert result.payment_method == "Tunai"

    def test_parse_alfamart_receipt(self):
        """Parse struk Alfamart"""
        raw_text = """
        ALFAMART
        JL GATOT SUBROTO 456
        BANDUNG
        
        Tgl: 20-11-2024 09:15
        
        POCARI SWEAT 500ML    5.500
        CHITATO 68GR          8.000
        
        Total Rp 13.500
        Bayar QRIS  13.500
        
        Terima Kasih
        """
        result = self.processor._parse_receipt(raw_text)
        
        assert result.receipt_type == ReceiptType.ALFAMART
        assert result.date == "2024-11-20"
        assert result.total == 13500.0
        assert result.payment_method == "QRIS"

    def test_parse_cafe_receipt(self):
        """Parse struk cafe"""
        raw_text = """
        STARBUCKS COFFEE
        GRAND INDONESIA
        
        06 Dec 2024 16:45
        
        CARAMEL MACCHIATO GRANDE    65.000
        CROISSANT                   35.000
        
        Subtotal                   100.000
        PB1 10%                     10.000
        
        Grand Total               110.000
        
        Payment: GOPAY
        
        Thank You!
        """
        result = self.processor._parse_receipt(raw_text)
        
        assert result.receipt_type == ReceiptType.CAFE
        assert "starbucks" in result.merchant.lower()
        assert result.date == "2024-12-06"
        assert result.total == 110000.0
        assert result.payment_method == "GoPay"


class TestEdgeCases:
    """Test edge cases dan error handling"""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_empty_text(self):
        """Empty text"""
        result = self.processor._parse_receipt("")
        assert result.total is None
        assert result.merchant is None

    def test_garbage_text(self):
        """Garbage/noise text"""
        result = self.processor._parse_receipt("asdfghjkl 12345 !@#$%")
        assert result.confidence < 0.5

    def test_partial_receipt(self):
        """Struk terpotong"""
        raw_text = """
        ...
        TOTAL    50.000
        TUNAI    50.000
        """
        result = self.processor._parse_receipt(raw_text)
        assert result.total == 50000.0

    def test_blurry_numbers(self):
        """Angka tidak jelas (OCR error)"""
        raw_text = """
        INDOMARET
        Total 5O.OOO
        """
        # Should still try to extract what it can
        result = self.processor._parse_receipt(raw_text)
        assert result.receipt_type == ReceiptType.INDOMARET


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
