from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import AppConfig
from app.db.session import init_db

app_config = AppConfig()
app = FastAPI(title=app_config.description, version=app_config.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()


@app.get("/")
async def root():
    return {"message": "backend-api up", "version": app_config.version}
