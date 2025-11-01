# backend-api

FastAPI lengkap untuk menangani business logic pencatatan keuangan WhatsApp.

Fitur yang tersedia:

- ORM SQLAlchemy async (PostgreSQL) dengan skema users/transactions/savings
- Endpoint transaksi, tabungan, laporan harian, dan webhook WA
- Integrasi layanan AI untuk parsing intent, STT, dan OCR
- Logging WA message + audit AI otomatis

## Menjalankan Lokal

```bash
cd services/backend-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Gunakan variabel `DATABASE_URL` untuk mengganti koneksi default `postgresql+asyncpg://postgres:postgres@db:5432/keuangan`.

### Migrasi Database

Backend otomatis menjalankan `alembic upgrade head` saat startup bila `AUTO_RUN_MIGRATIONS=true` (default). Untuk menjalankan manual:

```bash
cd services/backend-api
alembic upgrade head
```

Untuk membuat revisi baru:

```bash
alembic revision -m "deskripsi" --autogenerate
```

### Pengujian Migrasi

Tersedia serangkaian tes `pytest` yang memastikan migrasi Alembic bisa mencapai `head` dan struktur tabel sesuai ekspektasi. Siapkan terlebih dahulu database PostgreSQL uji, misalnya `keuangan_bot_test`, lalu jalankan:

```bash
export TEST_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/keuangan_bot_test"
pytest
```

Tes akan otomatis men-downgrade/upgrade migrasi pada database uji. Bila database belum tersedia atau tidak dapat dihubungi, seluruh tes akan diskip otomatis.

