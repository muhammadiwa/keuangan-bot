from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from loguru import logger

from app.core.config import get_settings

try:  # pragma: no cover - alembic optional during type checking
    from alembic import command
    from alembic.config import Config
except ModuleNotFoundError:  # pragma: no cover
    command = None
    Config = None

settings = get_settings()
engine = create_async_engine(settings.database_url, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Ensure database migrations are applied before serving traffic."""

    if not settings.auto_run_migrations or command is None or Config is None:
        return

    alembic_ini = Path(__file__).resolve().parents[2] / "alembic.ini"
    if not alembic_ini.exists():
        return

    config = Config(str(alembic_ini))
    try:
        await asyncio.to_thread(command.upgrade, config, "head")
        logger.info("Database migrations are up-to-date")
    except Exception:  # pragma: no cover - propagate for FastAPI startup failure
        logger.exception("Failed to apply database migrations")
        raise
