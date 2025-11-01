# Keuangan Bot Monorepo

Monorepo ini menampung seluruh service untuk aplikasi pencatatan keuangan pribadi berbasis WhatsApp yang dijelaskan pada `docs/whatsapp_ai_finance_spec.md`.

## Struktur

- `services/wa-bot-service` — Bot WhatsApp berbasis Baileys yang menangani sesi multi-user dan mengirimkan webhook ke backend.
- `services/backend-api` — FastAPI untuk business logic keuangan, laporan, dan dashboard.
- `services/ai-media-service` — FastAPI yang membungkus pipeline NLU, STT, dan OCR self-hosted.
- `docs/` — Spesifikasi dan dokumentasi pendukung.

Gunakan `docker-compose.yml` untuk menjalankan seluruh komponen secara lokal.

## Alur Branch

- Seluruh perubahan aktif kini hidup pada branch `master`.
- Saat melakukan pengembangan fitur baru, buat branch turunan lalu lakukan merge balik ke `master` setelah review.
- Untuk menyinkronkan perubahan lokal dengan branch utama jalankan `git checkout master && git pull` sebelum melakukan `docker compose up`.

