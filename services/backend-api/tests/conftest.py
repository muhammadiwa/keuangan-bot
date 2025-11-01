from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.config import get_settings

DEFAULT_ASYNC_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/keuangan_bot_test"


def _ensure_driver_pair(url: str) -> Dict[str, str]:
    """Return async/sync variants for the given database URL."""
    async_url = url
    if "+asyncpg" not in async_url and "+psycopg" in async_url:
        async_url = async_url.replace("+psycopg", "+asyncpg")
    elif "+asyncpg" not in async_url and "+psycopg" not in async_url:
        async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")

    if "+asyncpg" in async_url:
        sync_url = async_url.replace("+asyncpg", "+psycopg")
    else:
        sync_url = async_url

    return {"async": async_url, "sync": sync_url}


@pytest.fixture(scope="session")
def db_urls() -> Dict[str, str]:
    custom_url = os.getenv("TEST_DATABASE_URL", DEFAULT_ASYNC_URL)
    urls = _ensure_driver_pair(custom_url)

    # Refresh cached settings so Alembic picks the injected DATABASE_URL
    os.environ["DATABASE_URL"] = urls["async"]
    get_settings.cache_clear()
    get_settings()

    return urls


@pytest.fixture(scope="session")
def alembic_config(db_urls: Dict[str, str]) -> Config:
    cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_urls["async"])
    return cfg


@pytest.fixture(scope="session")
def engine(db_urls: Dict[str, str]) -> Engine:
    engine = create_engine(db_urls["sync"], future=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - best effort guard
        pytest.skip(f"Database test URL unavailable: {exc}")
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def migrated_engine(engine: Engine, alembic_config: Config) -> Engine:
    try:
        command.downgrade(alembic_config, "base")
    except Exception:
        # Database might already be clean; ignore errors for idempotency
        pass

    command.upgrade(alembic_config, "head")

    yield engine

    try:
        command.downgrade(alembic_config, "base")
    except Exception:
        pass
