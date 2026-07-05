from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import Base


def create_sqlite_engine(database_path: Path) -> Engine:
    """Create a SQLite engine for a local database file."""

    database_path.parent.mkdir(parents=True, exist_ok=True)

    return create_engine(
        f"sqlite:///{database_path.resolve()}",
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create sessions used by repositories."""

    return sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )


def create_schema(engine: Engine) -> None:
    """Create all currently defined database tables."""

    Base.metadata.create_all(engine)
