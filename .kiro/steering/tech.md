# Technology Stack

## Architecture
Microservices architecture dengan 3 service utama yang berkomunikasi via HTTP internal dalam Docker network.

## Core Services

### wa-bot-service (Node.js 20)
- **Framework**: Express.js
- **WhatsApp Library**: @whiskeysockets/baileys v6.6.0
- **Storage**: MinIO client untuk media files
- **Logging**: Pino
- **Session Management**: File-based WhatsApp sessions

### backend-api (Python 3.11)
- **Framework**: FastAPI 0.110.0
- **ASGI Server**: Uvicorn
- **Database**: PostgreSQL 15 dengan SQLAlchemy 2.0.29
- **Migration**: Alembic 1.13.1
- **HTTP Client**: httpx untuk service communication
- **Validation**: Pydantic

### ai-media-service (Python 3.11)
- **Framework**: FastAPI 0.110.0
- **STT**: faster-whisper 1.0.1
- **OCR**: pytesseract 0.3.10 + Pillow
- **NLU**: Ollama integration (phi3/llama3/qwen2.5 models)

## Infrastructure

### Database
- **Primary**: PostgreSQL 15
- **Cache/Queue**: Redis 7
- **Storage**: MinIO (S3-compatible)

### AI Models
- **NLU**: Ollama dengan model quantized (qwen2.5:1.5b, phi3:mini)
- **STT**: Whisper medium model via faster-whisper
- **OCR**: Tesseract dengan bahasa Indonesia + English

## Common Commands

### Development
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f [service-name]

# Rebuild specific service
docker compose build [service-name]
docker compose up -d [service-name]

# Database migration
docker compose exec backend-api alembic upgrade head

# Access database
docker compose exec db psql -U postgres -d keuangan
```

### Testing
```bash
# Backend API tests
docker compose exec backend-api pytest

# Check service health
curl http://localhost:8000/health
curl http://localhost:8500/health
curl http://localhost:3000/health
```

### Deployment
```bash
# Production deployment
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Backup database
docker compose exec db pg_dump -U postgres keuangan > backup.sql

# Sync branch before deployment
git checkout master && git pull
```

## Environment Configuration
Each service uses `.env` files for configuration. Copy from `.env.example` and adjust for local/production environments.

## Port Allocation
- wa-bot-service: 3000
- backend-api: 8000  
- ai-media-service: 8500
- ollama: 11434
- postgresql: 5432
- minio: 9000 (API), 9001 (Console)
- redis: 6379