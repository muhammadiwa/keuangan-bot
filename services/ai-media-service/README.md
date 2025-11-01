# ai-media-service

FastAPI service untuk membungkus NLU, STT, dan OCR menggunakan model open-source/self-host.

Fitur utama saat ini:

- integrasi Ollama (`POST /ai/parse`) dengan prompt JSON untuk intent keuangan
- transkripsi audio WhatsApp via `faster-whisper`
- OCR struk menggunakan `pytesseract`
- fallback heuristik ketika model tidak tersedia agar alur tidak terputus

## Pengembangan Lokal

```bash
cd services/ai-media-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8500
```

Sesuaikan `.env` untuk mengatur `OLLAMA_URL`, `OLLAMA_MODEL`, dan model Whisper/Tesseract yang tersedia di VPS.

