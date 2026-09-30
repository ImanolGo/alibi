"""Postgres + pgvector: SQLAlchemy models and session plumbing.

``create_all`` only (no Alembic), as per the plan. The engine is built lazily,
so importing this module never needs a database — only calling it does.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, String, Text, create_engine, func, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import EMBEDDING_DIM, get_settings


class Base(DeclarativeBase):
    pass


class Memory(Base):
    """One thing a suspect remembers, with the embedding of that text."""

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[str] = mapped_column(String(128), index=True)
    suspect_id: Mapped[str] = mapped_column(String(128), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "ix_memories_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    """Build (once per URL) the SQLAlchemy engine."""
    url = url or get_settings().database_url
    return create_engine(url, pool_pre_ping=True)


@lru_cache(maxsize=4)
def get_session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)


def init_db(engine: Engine | None = None) -> None:
    """Enable pgvector and create the tables if they do not exist yet."""
    engine = engine or get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
