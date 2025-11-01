# Spesifikasi Aplikasi Keuangan Pribadi Berbasis AI via WhatsApp

## Ringkasan
Aplikasi ini menyediakan pencatatan keuangan pribadi multi-user melalui bot WhatsApp Web (Baileys) dengan dukungan input teks, voice note, dan foto struk. Semua AI bersifat self-host (Ollama, Whisper, Tesseract) dan dapat dijalankan di VPS Ubuntu. Backend utama dibangun dengan **FastAPI** karena:

1. **Performa & Skalabilitas**: FastAPI berbasis ASGI, mendukung async I/O, cocok untuk integrasi dengan worker STT/OCR dan komunikasi HTTP internal.
2. **Ekosistem Python**: Memudahkan integrasi langsung dengan layanan AI (Whisper, Tesseract, Ollama) dan pipeline data.
3. **Produktivitas**: Dukungan pydantic untuk validasi payload dan dokumentasi otomatis OpenAPI.

## Arsitektur Sistem
Semua komponen dapat dijalankan via Docker Compose dan berkomunikasi melalui jaringan internal.

| Service              | Bahasa/Stack         | Port | Deskripsi                                                                 |
|----------------------|----------------------|------|----------------------------------------------------------------------------|
| `wa-bot-service`     | Node.js 20 + Baileys | 3000 | Menangani sesi WhatsApp Web (QR/pairing), menerima & mengirim pesan.      |
| `backend-api`        | FastAPI (Python 3.11)| 8000 | Business logic keuangan, webhook WA, manajemen tabungan, laporan, auth.   |
| `ai-media-service`   | Python 3.11          | 8500 | Endpoint NLU, STT, OCR. Berinteraksi dengan Ollama (11434), Whisper, dll. |
| `ollama`             | Ollama runtime       | 11434| Menjalankan model NLU (phi3/llama3/qwen2.5 1.5B).                          |
| `minio`              | MinIO                | 9000 | Penyimpanan media audio/gambar (S3 compatible).                            |
| `db`                 | PostgreSQL 15        | 5432 | Penyimpanan relasional transaksi, tabungan, audit.                         |
| `redis` (opsional)   | Redis                | 6379 | Queue untuk job async (STT/OCR) dan rate limiting.                         |

### Komunikasi Internal
- `wa-bot-service` menerima event WA, lalu POST ke `backend-api /wa/incoming`.
- `backend-api` menyimpan raw message, memutuskan langkah lanjutan (NLU, STT, OCR) dengan memanggil `ai-media-service`.
- `ai-media-service` menjalankan pipeline AI (panggil Ollama/Whisper/Tesseract) dan mengembalikan hasil struktur.
- `backend-api` menyimpan transaksi/tabungan, lalu menginstruksikan `wa-bot-service` melalui `POST /deliver` untuk mengirim respon ke pengguna.
- File media dari WA disimpan di MinIO, referensi URL private digunakan oleh layanan AI.

## Detail Tiap Service
### 1. wa-bot-service (Node.js 20 + Baileys)
- Login menggunakan QR atau pairing code.
- Menangani multi-user session dengan penyimpanan session file per nomor.
- Fitur utama:
  - Menangkap pesan masuk (text/audio/image/document), mengunduh media.
  - Menyimpan media ke MinIO melalui Signed PUT.
  - Log setiap pesan dan ack.
  - Menyediakan endpoint internal:
    - `GET /session/:id/qr` → Mengambil QR base64.
    - `POST /deliver` → Mengirim pesan teks/template ke user.
  - Forward payload terstruktur ke `backend-api` dengan minimal field: nomor pengirim, tipe pesan, teks, URL media, timestamp.

### 2. backend-api (FastAPI)
- Fitur:
  - Endpoint `POST /wa/incoming` untuk memproses semua pesan WA.
  - Handler intent transaksi/tabungan/laporan.
  - CRUD transaksi & tabungan.
  - Scheduler (APScheduer/Celery beat) untuk laporan harian/mingguan/bulanan.
  - Autentikasi JWT untuk dashboard web.
  - Dashboard (FastAPI + Jinja2/HTMX) menampilkan transaksi, filter, ekspor CSV.
  - Logging ke tabel `wa_messages` dan `ai_audit`.
- Integrasi
  - Panggil `ai-media-service` sesuai tipe media.
  - Gunakan SQLAlchemy + PostgreSQL.
  - Upload/download media dari MinIO.

### 3. ai-media-service (Python)
- Menyediakan API:
  - `POST /nlu/parse-text` → Panggil Ollama model (phi3/llama3/qwen2.5 1.5B) dengan prompt khusus.
  - `POST /media/stt` → Worker async memanggil Whisper/faster-whisper pada file audio dari MinIO.
  - `POST /media/ocr` → Jalankan Tesseract (atau PaddleOCR jika diperlukan) pada gambar struk.
- Menyimpan trace ke `ai_audit` via webhook/queue.
- Dijalankan sebagai layanan async (FastAPI + Celery worker) untuk heavy jobs.

### 4. Database (PostgreSQL)
- Semua transaksi, tabungan, pesan WA, audit AI tersimpan.
- Gunakan migration tool (Alembic) dari backend.

### 5. Storage (MinIO)
- Bucket: `wa-media` (untuk audio/image), `reports` (untuk file laporan).
- Gunakan presigned URL untuk akses antar layanan.

### 6. Dashboard Web
- Modul pada `backend-api`.
- Rute: `/login`, `/dashboard`, `/transactions`, `/savings`, `/reports/download`.
- Gunakan server-side rendering ringan (Jinja2) + chart (Chart.js) untuk visual.

## Desain Database
```sql
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    phone VARCHAR(20) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    timezone VARCHAR(64) DEFAULT 'Asia/Jakarta',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_users_phone ON users(phone);

CREATE TABLE categories (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(80) NOT NULL,
    keywords TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_categories_user ON categories(user_id);
CREATE UNIQUE INDEX idx_categories_user_name ON categories(user_id, name);

CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    direction VARCHAR(20) NOT NULL CHECK (direction IN ('income','expense','transfer')),
    amount NUMERIC(18,2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'IDR',
    category_id BIGINT REFERENCES categories(id),
    description TEXT,
    tx_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    source VARCHAR(20) NOT NULL CHECK (source IN ('whatsapp','web','import')),
    raw_text TEXT,
    ai_confidence NUMERIC(5,2),
    status VARCHAR(20) NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed','pending')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_transactions_user_datetime ON transactions(user_id, tx_datetime);
CREATE INDEX idx_transactions_category ON transactions(category_id);
CREATE INDEX idx_transactions_direction ON transactions(direction);

CREATE TABLE savings_accounts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    target_amount NUMERIC(18,2) NOT NULL,
    current_amount NUMERIC(18,2) NOT NULL DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'IDR',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_savings_user_name ON savings_accounts(user_id, name);

CREATE TABLE savings_transactions (
    id BIGSERIAL PRIMARY KEY,
    savings_account_id BIGINT NOT NULL REFERENCES savings_accounts(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    direction VARCHAR(20) NOT NULL CHECK (direction IN ('deposit','withdraw')),
    amount NUMERIC(18,2) NOT NULL,
    note TEXT,
    source VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_savings_tx_account ON savings_transactions(savings_account_id);
CREATE INDEX idx_savings_tx_user ON savings_transactions(user_id);

CREATE TABLE wa_messages (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    wa_from VARCHAR(20) NOT NULL,
    wa_body TEXT,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('in','out')),
    intent VARCHAR(50),
    response TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_wa_messages_user ON wa_messages(user_id);
CREATE INDEX idx_wa_messages_from_created ON wa_messages(wa_from, created_at DESC);

CREATE TABLE ai_audit (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    model_name VARCHAR(120) NOT NULL,
    raw_input TEXT NOT NULL,
    raw_output TEXT,
    success BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_ai_audit_user_created ON ai_audit(user_id, created_at DESC);
```

## Spesifikasi API
### 1. `POST /wa/incoming`
- **Deskripsi**: Menerima pesan dari `wa-bot-service`, menyimpan log, menentukan tindakan.
- **Request Body**
```json
{
  "from_number": "6281234567890",
  "message_type": "text|audio|image",
  "text": "beli pulsa 50k",
  "media_url": null,
  "timestamp": 1730368902
}
```
- **Response (200)**
```json
{
  "reply": "✅ Catat pengeluaran Rp50.000 kategori Pulsa & Data. Balas: ganti <kategori> kalau mau ubah.",
  "status": "ok"
}
```

### 2. `POST /ai/parse`
- **Deskripsi**: Wrapper ke `ai-media-service /nlu/parse-text`.
- **Request**
```json
{ "text": "beli pulsa 50k" }
```
- **Response**
```json
{
  "intent": "create_transaction",
  "amount": 50000,
  "currency": "IDR",
  "description": "beli pulsa",
  "category_suggestion": "Pulsa & Data",
  "datetime": "2024-05-01T08:30:00+07:00"
}
```

### 3. `POST /media/stt`
- **Request**
```json
{ "media_url": "https://minio.local/wa-media/abc123.ogg" }
```
- **Response**
```json
{ "text": "tadi makan di luar tiga puluh lima ribu" }
```

### 4. `POST /media/ocr`
- **Request**
```json
{ "media_url": "https://minio.local/wa-media/struk-xyz.jpg" }
```
- **Response**
```json
{
  "total": 132500,
  "date": "2025-10-16",
  "merchant": "Indomaret",
  "raw_text": "INDOMARET\nTotal 132.500\n..."
}
```

### 5. `GET /reports/daily`
- **Query**: `?user_id=...&date=2024-05-01`
- **Response**
```json
{
  "date": "2024-05-01",
  "total_income": 3500000,
  "total_expense": 250000,
  "category_breakdown": [
    { "category": "Makan", "amount": 150000 },
    { "category": "Transport", "amount": 100000 }
  ],
  "month_balance": 3250000
}
```

### 6. `POST /savings/accounts`
- **Request**
```json
{ "name": "laptop", "target_amount": 15000000 }
```
- **Response**
```json
{
  "id": 12,
  "name": "laptop",
  "target_amount": 15000000,
  "current_amount": 0,
  "currency": "IDR"
}
```

### 7. `POST /savings/deposit`
- **Request**
```json
{ "saving_name": "laptop", "amount": 500000 }
```
- **Response**
```json
{
  "message": "✅ Setoran Rp500.000 ke tabungan laptop. Progress 3%."
}
```

### Endpoint Tambahan
- `POST /savings/withdraw`
- `GET /transactions?user_id=&from=&to=&category=`
- `POST /transactions`
- `GET /reports/weekly`, `GET /reports/monthly`
- `POST /wa/send` (internal) untuk memicu pengiriman manual.

## Flow WhatsApp
### Flow 1 – Text
1. **WA User → WA Bot**: Kirim teks "beli bensin 50k".
2. **WA Bot → Backend (`/wa/incoming`)**: Payload text + metadata.
3. **Backend → AI (`/nlu/parse-text`)**: Deteksi intent & entitas.
4. **AI → Backend**: Kembalikan JSON transaksi.
5. **Backend**: Simpan transaksi, update kategori, log `wa_messages`.
6. **Backend → WA Bot (`/deliver`)**: Kirim pesan konfirmasi.
7. **WA Bot → WA User**: Kirim "✅ Catat pengeluaran...".

### Flow 2 – Voice
1. **WA User → WA Bot**: Kirim voice note OGG.
2. **WA Bot**: Download audio, simpan ke MinIO, kirim payload ke backend.
3. **Backend → AI (`/media/stt`)**: Konversi audio ke teks.
4. **AI → Backend**: Berikan teks transkrip.
5. **Backend → AI (`/nlu/parse-text`)**: Ekstraksi intent.
6. **Backend**: Simpan transaksi + log.
7. **Backend → WA Bot (`/deliver`)**: Kirim konfirmasi "aku dengar: ...".
8. **WA Bot → WA User**: Kirim pesan.

### Flow 3 – Gambar/Struk
1. **WA User → WA Bot**: Kirim foto struk.
2. **WA Bot**: Download, unggah ke MinIO, kirim payload ke backend.
3. **Backend → AI (`/media/ocr`)**: Jalankan OCR.
4. **AI → Backend**: Hasil total, tanggal, merchant, raw text.
5. **Backend → AI (`/nlu/parse-text`)**: Gunakan ringkasan struk untuk tentukan kategori.
6. **Backend**: Simpan transaksi, status pending jika kepercayaan rendah, log.
7. **Backend → WA Bot**: Kirim konfirmasi + opsi edit kategori.
8. **WA Bot → WA User**: Kirim pesan.

### Flow 4 – Laporan Harian
1. **Cron (APScheduer/Celery Beat)**: Jam 21:00 WIB trigger.
2. **Backend**: Ambil data transaksi harian, format laporan.
3. **Backend → WA Bot**: Panggil `/deliver` untuk setiap user aktif.
4. **WA Bot → WA User**: Kirim laporan ringkas via WA.

## AI Layer Detail
### 1. Prompt NLU Transaksi Teks
```
Tugas Anda mendeteksi intent keuangan dari pesan pengguna (bahasa Indonesia informal).
Outputkan JSON ketat tanpa teks lain dengan format:
{"intent": ..., "amount": ..., "currency": "IDR", "description": ..., "category_suggestion": ..., "datetime": ..., "notes": ...}
Intent yang valid: create_transaction, create_saving, deposit_saving, withdraw_saving, get_report, set_budget, smalltalk.
Jika tidak ada angka uang, amount null. Gunakan format datetime ISO8601 zona Asia/Jakarta.
Kategorikan sesuai kosakata umum (Makan, Transport, Pulsa & Data, Gaji, dll).
```

### 2. Prompt OCR Struk (post-processing)
```
Rapikan hasil OCR berikut agar menjadi ringkasan JSON ketat:
{"total": <angka dalam IDR>, "date": "YYYY-MM-DD", "merchant": "...", "items": [ {"name": ..., "price": ...}, ... ]}
Jika tanggal tidak ditemukan, gunakan tanggal hari ini.
```

### 3. Prompt Tabungan
```
Deteksi perintah tabungan. Jika ada kata seperti "tabungan", "nabung", "setor", "tarik", "saving", tentukan intent:
- buat tabungan → create_saving (extract target_amount jika ada)
- setor → deposit_saving
- tarik → withdraw_saving
Jawab dalam JSON ketat sebagaimana format utama.
```

### 4. Prompt Klarifikasi Ambigu
```
Jika pesan ambigu (tidak ada kategori/amount jelas), set intent=clarification dan sertakan field "question" untuk ditanyakan ke user.
```

### Model AI
- **NLU**: Ollama model `phi3:mini` atau `llama3:instruct` dengan quantization `q4_0` agar hemat RAM.
- **STT**: `faster-whisper` model `medium` dijalankan via worker GPU opsional atau CPU (use INT8).
- **OCR**: Tesseract bahasa Indonesia + English (`tesseract-ocr-ind`) dengan optional PaddleOCR untuk fallback.

## Non-Functional Requirements
- **Kinerja**: STT & OCR diproses async menggunakan Celery worker (Redis sebagai broker). Backend mengembalikan respon awal cepat, update WA ketika selesai.
- **Keamanan**: Nomor baru diverifikasi (buat user otomatis tapi status pending). Hanya user terdaftar yang dapat auto-catatan; lainnya minta konfirmasi identitas. JWT + hashed password (Argon2) untuk dashboard.
- **Observability**: Gunakan structured logging (JSON) di semua service, kirim ke Loki/ELK opsional. `wa_messages` menyimpan seluruh percakapan in/out.
- **Storage**: Semua media WA disimpan di MinIO dengan lifecycle policy. Metadata (URL, hash) disimpan pada tabel `wa_messages` atau tabel terpisah `media_assets` jika diperlukan.
- **Deployment**: Docker Compose dengan network internal. Gunakan Traefik/Nginx untuk reverse proxy HTTPS. Sertakan healthcheck di setiap service.
- **Backup & Recovery**: Snapshot PostgreSQL rutin, MinIO mirror ke storage lain.

## Jadwal Cron & Otomasi
- Harian (21:00 WIB): Kirim laporan harian.
- Mingguan (Setiap Senin 08:00 WIB): Laporan mingguan.
- Bulanan (Setiap 1 tanggal 08:00 WIB): Laporan bulan sebelumnya.
- Reminder tabungan (opsional): Jika progress < target pada tanggal tertentu.

## Dashboard Web – Fitur Detail
- **Login**: email/phone + password.
- **Transaksi**: Tabel paginated, filter by tanggal, kategori, sumber, search deskripsi.
- **Grafik**: Bar chart pengeluaran per kategori, line chart cashflow.
- **Export**: Tombol `Export CSV` memanggil `GET /transactions/export`.
- **Savings**: Menampilkan list tabungan, progress bar.

## Penanganan Multi-User
- Identifikasi user via `from_number`. Jika belum ada user, buat `users` baru dengan default name `User <last4>` dan timezone Asia/Jakarta.
- Mapping WA session ID → user_id disimpan di `users.phone`.
- Setiap pesan dicatat di `wa_messages` dengan `direction='in'`. Balasan disimpan `direction='out'`.

## Pseudocode Alur `/wa/incoming`
```python
@app.post("/wa/incoming")
async def wa_incoming(payload: IncomingMessage):
    user = get_or_create_user(payload.from_number)
    log_message(user, payload)

    if payload.message_type == "audio":
        text = await ai_service.stt(payload.media_url)
        append_ai_audit(user, "whisper", payload.media_url, text)
        nlu_result = await ai_service.parse_text(text)
    elif payload.message_type == "image":
        ocr = await ai_service.ocr(payload.media_url)
        nlu_result = await ai_service.parse_text(
            format_receipt_text(ocr, payload.text)
        )
    else:
        nlu_result = await ai_service.parse_text(payload.text)

    response = handle_intent(user, nlu_result, payload)
    save_response(user, response)
    await wa_bot.deliver(user.phone, response.message)
    return {"status": "ok", "reply": response.message}
```

## Test Case (Minimal 10)
1. **"beli pulsa 50k"** → intent `create_transaction`, expense, kategori "Pulsa & Data".
2. **"gaji 3,5jt dari kantor"** → intent `create_transaction`, income, kategori "Gaji".
3. **"setor laptop 500k"** → intent `deposit_saving`, tabungan `laptop`, amount 500000.
4. **"laporan hari ini"** → intent `get_report`, trigger `GET /reports/daily`.
5. **Voice note**: "tadi makan di luar 35 ribu" → STT + create expense kategori "Makan".
6. **Foto struk Indomaret** dengan total 132.500 → OCR + create expense pending.
7. **"buat tabungan haji 25 juta"** → intent `create_saving`, target 25000000.
8. **"tarik laptop 200k"** → intent `withdraw_saving`, nominal 200000.
9. **"pengeluaran makan bulan ini berapa"** → intent `get_report` dengan filter kategori.
10. **Pesan ambigu**: "tadi 50k" → intent `clarification`, bot menanyakan "Ini pengeluaran kategori apa?".
11. **Smalltalk**: "makasih ya" → intent `smalltalk`, balasan ramah tanpa catat transaksi.
12. **Multi-user**: Nomor baru mengirim "gaji 5jt" → sistem buat user baru otomatis dan catat income.

## Contoh Payload Balasan WA
- Sukses transaksi: "✅ Catat pengeluaran Rp50.000 kategori Transport. Balas: ganti <kategori> kalau mau ubah."
- Konfirmasi OCR: "Aku baca struk Indomaret total Rp132.500 tanggal 16-10-2025. ✅ Sudah dicatat sebagai pengeluaran. Balas: ganti <kategori> untuk koreksi."
- Tabungan: "✅ Setoran Rp500.000 ke tabungan laptop. Progress 500.000/15.000.000 (3%)."
- Laporan harian: "Ringkasan 21 Jan 2025\nIncome: Rp3.500.000\nExpense: Rp250.000\nTop kategori: Makan Rp150.000, Transport Rp100.000\nSaldo bulan ini: Rp3.250.000."

## Contoh Docker Compose (Ringkas)
```yaml
services:
  wa-bot:
    build: ./services/wa-bot
    ports:
      - "3000:3000"
    environment:
      - BACKEND_URL=http://backend:8000
    volumes:
      - ./data/wa-sessions:/app/sessions

  backend:
    build: ./services/backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://bot:secret@db:5432/keuangan
      - MINIO_ENDPOINT=http://minio:9000
      - AI_SERVICE_URL=http://ai:8500
    depends_on:
      - db
      - ai

  ai:
    build: ./services/ai-media
    ports:
      - "8500:8500"
    environment:
      - OLLAMA_URL=http://ollama:11434

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ./data/ollama:/root/.ollama

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=bot
      - POSTGRES_PASSWORD=secret
      - POSTGRES_DB=keuangan
    volumes:
      - ./data/postgres:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    environment:
      - MINIO_ROOT_USER=minio
      - MINIO_ROOT_PASSWORD=miniopass
    command: server /data
    ports:
      - "9000:9000"
```

## Rencana Pengujian & Monitoring
- Unit test untuk parser intent, handler tabungan, formatter laporan.
- Integration test memalsukan webhook WA dan memastikan transaksi tersimpan.
- Load test untuk 100 pesan serentak (kuy dengan k6).
- Monitoring: healthcheck `/healthz` pada tiap service, metrics Prometheus (fastapi instrumentation).

## Proses Onboarding User Baru
1. Nomor baru mengirim pesan.
2. Backend membuat user dengan default `name`.
3. Kirim pesan sambutan + panduan: "Halo! Aku bot keuanganmu. Contoh: 'beli kopi 20k'."
4. Dorong user membuat password via link dashboard (token one-time).

## Mekanisme Klarifikasi & Edit
- Jika user mengetik `ganti <kategori>` setelah konfirmasi, backend update `category_id` transaksi terakhir (by `wa_messages` context).
- Perintah `hapus transaksi terakhir` → status `deleted` (opsional, bisa field tambahan `is_deleted`).

## Pertimbangan Resource VPS
- Minimal 4 vCPU, 8GB RAM.
- Jalankan Ollama model kecil (`qwen2.5:1.5b` quantized) ~3GB.
- Worker STT/OCR bisa scale horizontal.
- Gunakan streaming chunk untuk download media WA agar hemat memori.

## Roadmap Pengembangan
1. Fase 1: Setup WA bot + backend basic text intent.
2. Fase 2: Tambah STT & OCR worker, integrasi MinIO.
3. Fase 3: Dashboard web & laporan otomatis.
4. Fase 4: Optimasi AI, kustom kategori, multi-language.
5. Fase 5: Integrasi budget & alert keuangan.


## Summary Eksekusi & Next Steps
**Summary:**
- Spesifikasi arsitektur multi-service (WA bot, backend, AI media) dengan detail komunikasi dan deployment lokal.
- Skema database relasional lengkap beserta indeks untuk transaksi, tabungan, log WA, dan audit AI.
- Kontrak API, flow WhatsApp, prompt AI, non-fungsional, serta daftar test case untuk input teks, suara, dan struk.

**Next Steps:**
1. Siapkan repository monorepo dengan struktur service (`wa-bot-service`, `backend-api`, `ai-media-service`).
2. Implementasikan `backend-api` FastAPI beserta model database dan migrasi awal sesuai skema.
3. Integrasikan `wa-bot-service` (Baileys) dengan endpoint `/wa/incoming` dan storage MinIO.
4. Bangun worker STT/OCR di `ai-media-service` dengan koneksi Ollama, Whisper, dan Tesseract.
5. Rancang dashboard web dasar (FastAPI + templating atau frontend ringan) dan endpoint laporan.
6. Konfigurasi pipeline deployment Docker Compose dan tambahkan monitoring (Prometheus/Grafana opsional).
