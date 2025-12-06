# Konfigurasi Storage - Keuangan Bot

## Overview

wa-bot-service mendukung dua jenis storage untuk menyimpan media (audio, gambar, dokumen):

| Storage Type | Deskripsi | Use Case |
|--------------|-----------|----------|
| **MinIO** | S3-compatible object storage | Production, scalable, distributed |
| **Local** | Local filesystem | Development, single server, simple setup |

## Konfigurasi

### Environment Variables

```bash
# Pilih storage type: 'minio' atau 'local'
STORAGE_TYPE=minio
```

### MinIO Storage (Default)

```bash
STORAGE_TYPE=minio

# MinIO Configuration
MINIO_ENDPOINT=http://minio:9000
MINIO_REGION=us-east-1
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=miniopass
MEDIA_BUCKET=wa-media
MINIO_PUBLIC_URL=http://minio:9000/wa-media
```

**Kelebihan:**
- Scalable untuk production
- S3-compatible API
- Built-in redundancy
- Web console untuk management
- Presigned URLs untuk security

**Kekurangan:**
- Butuh service tambahan (MinIO container)
- Lebih kompleks untuk setup

### Local Storage

```bash
STORAGE_TYPE=local

# Local Storage Configuration
LOCAL_STORAGE_PATH=./data/media
LOCAL_STORAGE_URL=http://localhost:3000/media
```

**Kelebihan:**
- Simple setup
- Tidak butuh service tambahan
- Mudah untuk development
- Direct filesystem access

**Kekurangan:**
- Tidak scalable untuk multi-server
- Tidak ada built-in redundancy
- Perlu backup manual

## Struktur File

### MinIO
```
minio-bucket/
├── wa-media/
│   ├── audio/
│   │   └── 2024-12-06T10-30-00-uuid.ogg
│   ├── images/
│   │   └── 2024-12-06T10-30-00-uuid.jpg
│   └── documents/
│       └── 2024-12-06T10-30-00-uuid.pdf
```

### Local Storage
```
data/media/
├── audio/
│   └── 2024-12-06T10-30-00-uuid.ogg
├── images/
│   └── 2024-12-06T10-30-00-uuid.jpg
└── documents/
    └── 2024-12-06T10-30-00-uuid.pdf
```

## API Endpoints

### Health Check
```bash
GET /healthz

# Response includes storage type
{
  "status": "ok",
  "session": ["default"],
  "storage": "local"  # atau "minio"
}
```

### Storage Info
```bash
GET /storage/info

# Response untuk local storage
{
  "type": "local",
  "config": {
    "basePath": "./data/media",
    "baseUrl": "http://localhost:3000/media"
  }
}

# Response untuk MinIO
{
  "type": "minio",
  "config": {
    "endpoint": "http://minio:9000",
    "bucket": "wa-media"
  }
}
```

### Media Access (Local Storage Only)
```bash
GET /media/{folder}/{filename}

# Example
GET /media/images/2024-12-06T10-30-00-abc123.jpg
```

## Docker Compose

### Dengan MinIO (Default)
```yaml
services:
  wa-bot-service:
    # ...
    volumes:
      - ./services/wa-bot-service/sessions:/app/sessions
    depends_on:
      - minio

  minio:
    image: minio/minio:latest
    # ...
```

### Dengan Local Storage
```yaml
services:
  wa-bot-service:
    # ...
    environment:
      - STORAGE_TYPE=local
      - LOCAL_STORAGE_PATH=/app/data/media
      - LOCAL_STORAGE_URL=http://wa-bot-service:3000/media
    volumes:
      - ./services/wa-bot-service/sessions:/app/sessions
      - ./data/media:/app/data/media  # Mount local storage
    # Tidak perlu depends_on minio
```

## Migrasi Storage

### Dari MinIO ke Local
1. Download semua file dari MinIO bucket
2. Copy ke `LOCAL_STORAGE_PATH`
3. Update `.env` dengan `STORAGE_TYPE=local`
4. Restart service

```bash
# Download dari MinIO (menggunakan mc client)
mc cp --recursive minio/wa-media/ ./data/media/

# Update .env
STORAGE_TYPE=local
LOCAL_STORAGE_PATH=./data/media

# Restart
docker compose restart wa-bot-service
```

### Dari Local ke MinIO
1. Pastikan MinIO running
2. Upload file ke MinIO bucket
3. Update `.env` dengan `STORAGE_TYPE=minio`
4. Restart service

```bash
# Upload ke MinIO
mc cp --recursive ./data/media/ minio/wa-media/

# Update .env
STORAGE_TYPE=minio

# Restart
docker compose restart wa-bot-service
```

## Troubleshooting

### Local Storage: Permission Denied
```bash
# Pastikan folder memiliki permission yang benar
chmod -R 755 ./data/media
chown -R 1000:1000 ./data/media  # Sesuaikan dengan user container
```

### Local Storage: File Not Found
- Pastikan `LOCAL_STORAGE_URL` sesuai dengan URL yang bisa diakses
- Untuk Docker, gunakan nama service: `http://wa-bot-service:3000/media`
- Untuk development lokal: `http://localhost:3000/media`

### MinIO: Connection Refused
- Pastikan MinIO container running
- Check network connectivity antar container
- Verify `MINIO_ENDPOINT` menggunakan nama service Docker

### MinIO: Access Denied
- Verify `MINIO_ACCESS_KEY` dan `MINIO_SECRET_KEY`
- Check bucket policy di MinIO console

## Best Practices

### Development
```bash
# Gunakan local storage untuk development
STORAGE_TYPE=local
LOCAL_STORAGE_PATH=./data/media
LOCAL_STORAGE_URL=http://localhost:3000/media
```

### Production
```bash
# Gunakan MinIO untuk production
STORAGE_TYPE=minio
MINIO_ENDPOINT=http://minio:9000
# ... konfigurasi MinIO lainnya

# Atau gunakan AWS S3
STORAGE_TYPE=minio
MINIO_ENDPOINT=https://s3.amazonaws.com
MINIO_REGION=ap-southeast-1
MINIO_ACCESS_KEY=your-aws-access-key
MINIO_SECRET_KEY=your-aws-secret-key
MEDIA_BUCKET=your-bucket-name
MINIO_PUBLIC_URL=https://your-bucket.s3.ap-southeast-1.amazonaws.com
```

### Backup

**Local Storage:**
```bash
# Backup
tar -czvf media-backup-$(date +%Y%m%d).tar.gz ./data/media/

# Restore
tar -xzvf media-backup-20241206.tar.gz -C ./
```

**MinIO:**
```bash
# Backup menggunakan mc
mc mirror minio/wa-media/ ./backup/media/

# Atau snapshot MinIO data volume
docker run --rm -v keuangan-bot_minio-data:/data -v $(pwd):/backup \
  alpine tar -czvf /backup/minio-backup.tar.gz /data
```
