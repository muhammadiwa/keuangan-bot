# Project Structure

## Monorepo Organization

```
keuangan-bot/
├── services/                    # Microservices
│   ├── wa-bot-service/         # WhatsApp bot (Node.js)
│   ├── backend-api/            # Main API (FastAPI)
│   └── ai-media-service/       # AI processing (FastAPI)
├── docs/                       # Documentation
├── data/                       # Persistent data (gitignored)
├── .github/                    # GitHub workflows
├── .kiro/                      # Kiro configuration
└── docker-compose.yml          # Orchestration

```

## Service Structure Patterns

### wa-bot-service/
```
src/
├── index.js                    # Entry point
├── whatsapp/                   # WhatsApp logic
├── handlers/                   # Message handlers
├── utils/                      # Utilities
└── config/                     # Configuration
sessions/                       # WhatsApp sessions (persistent)
```

### backend-api/
```
app/
├── main.py                     # FastAPI app
├── api/                        # API routes
│   ├── wa/                     # WhatsApp webhooks
│   ├── transactions/           # Transaction CRUD
│   ├── savings/                # Savings management
│   └── reports/                # Report generation
├── models/                     # SQLAlchemy models
├── services/                   # Business logic
├── core/                       # Core utilities
└── db/                         # Database utilities
alembic/                        # Database migrations
tests/                          # Test suite
```

### ai-media-service/
```
app/
├── main.py                     # FastAPI app
├── api/                        # API endpoints
│   ├── nlu/                    # Natural Language Understanding
│   ├── stt/                    # Speech-to-Text
│   └── ocr/                    # Optical Character Recognition
├── services/                   # AI service implementations
├── models/                     # Pydantic models
└── utils/                      # Utilities
```

## File Naming Conventions

### Python Files
- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

### JavaScript Files
- **Files**: `camelCase.js` or `kebab-case.js`
- **Classes**: `PascalCase`
- **Functions/Variables**: `camelCase`
- **Constants**: `UPPER_SNAKE_CASE`

## Configuration Files

### Environment Files
- `.env.example` - Template dengan default values
- `.env` - Local environment (gitignored)
- Gunakan pydantic-settings untuk Python services
- Gunakan dotenv untuk Node.js services

### Docker Files
- `Dockerfile` - Multi-stage builds preferred
- `docker-compose.yml` - Development setup
- `docker-compose.prod.yml` - Production overrides

## Database Organization

### Migration Files (Alembic)
- Format: `YYYY_MM_DD_HHMM_description.py`
- Selalu review migration sebelum apply
- Backup database sebelum migration production

### Model Organization
```python
# models/
├── __init__.py
├── user.py                     # User model
├── transaction.py              # Transaction model
├── savings.py                  # Savings models
└── audit.py                    # Audit/logging models
```

## API Organization

### Route Structure
- `/api/v1/` prefix untuk versioning
- RESTful conventions untuk CRUD operations
- Webhook endpoints di `/webhooks/`
- Health checks di `/health`

### Response Patterns
```python
# Success response
{"status": "success", "data": {...}}

# Error response  
{"status": "error", "message": "...", "code": "..."}

# WhatsApp response
{"reply": "...", "status": "ok"}
```

## Development Workflow

### Branch Strategy
- `master` - Main development branch
- Feature branches: `feature/description`
- Hotfix branches: `hotfix/description`
- Merge ke master setelah review

### Code Organization
- Satu concern per file/module
- Dependency injection untuk services
- Async/await untuk I/O operations
- Proper error handling dan logging

### Testing Structure
```
tests/
├── unit/                       # Unit tests
├── integration/                # Integration tests
├── fixtures/                   # Test data
└── conftest.py                 # Pytest configuration
```