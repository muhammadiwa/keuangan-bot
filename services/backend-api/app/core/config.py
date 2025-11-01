from functools import lru_cache
from pydantic import AnyHttpUrl, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "backend-api"
    database_url: str = "postgresql+asyncpg://postgres:postgres@db:5432/keuangan"
    ai_service_url: AnyHttpUrl | str = "http://localhost:8500"
    minio_endpoint: AnyHttpUrl | str = "http://localhost:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "miniopass"
    timezone: str = "Asia/Jakarta"
    jwt_secret: str = "change-me"
    auto_run_migrations: bool = True

    def get_sync_database_url(self) -> str:
        """Return a synchronous driver URL for Alembic/CLI usage."""

        if "+asyncpg" in self.database_url:
            return self.database_url.replace("+asyncpg", "+psycopg")
        return self.database_url


class AppConfig(BaseModel):
    version: str = "0.1.0"
    description: str = (
        "Backend API FastAPI untuk pencatatan keuangan via WhatsApp sesuai spesifikasi."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
