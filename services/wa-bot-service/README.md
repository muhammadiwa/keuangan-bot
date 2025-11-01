# wa-bot-service

Service Node.js 20 yang mengelola sesi WhatsApp Web menggunakan Baileys. Versi ini sudah:

- menjalankan autentikasi multi-file (QR & pairing code)
- mengunggah media masuk (audio/gambar) ke MinIO/S3
- meneruskan pesan ke backend API sesuai spesifikasi
- menyediakan endpoint `/deliver` untuk mengirim pesan ke pengguna

## Pengembangan Lokal

```bash
cd services/wa-bot-service
npm install
npm run dev
```

Siapkan variabel lingkungan minimal berikut:

```
BACKEND_API_URL=http://localhost:8000
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=miniopass
MINIO_PUBLIC_URL=http://localhost:9000/wa-media
MEDIA_BUCKET=wa-media
SESSION_ID=default
```

Endpoint `GET /session/:id/qr` dapat digunakan untuk mengambil QR code sementara. Semua pesan yang diterima otomatis diteruskan ke backend, sedangkan media disimpan di bucket MinIO untuk diproses layanan AI.

