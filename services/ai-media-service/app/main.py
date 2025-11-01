from fastapi import FastAPI

from app.config import get_settings
from app.routes import router

settings = get_settings()
app = FastAPI(title="ai-media-service", version="0.1.0")
app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "ai-media-service up",
        "whisper_model": settings.whisper_model,
        "tesseract_lang": settings.tesseract_lang,
    }
