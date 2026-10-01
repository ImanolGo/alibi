"""Per-suspect memory (RAG) backed by pgvector.

The store sits behind a tiny protocol so unit tests can fake it; production uses
:class:`SqlMemoryStore`. Every embedding comes from ``llm.embed`` — no model
call happens anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select

from .db import Memory, get_session_factory
from .llm import LLM, default_client


@dataclass(frozen=True)
class MemoryHit:
    """One recalled memory. ``score`` is cosine distance (smaller = closer)."""

    content: str
    score: float


class MemoryStore(Protocol):
    def add(
        self, *, game_id: str, suspect_id: str, content: str, embedding: list[float]
    ) -> None: ...

    def nearest(
        self, *, game_id: str, suspect_id: str, embedding: list[float], k: int
    ) -> list[MemoryHit]: ...


class SqlMemoryStore:
    """Postgres/pgvector-backed memory store."""

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def add(self, *, game_id: str, suspect_id: str, content: str, embedding: list[float]) -> None:
        with self._session_factory() as session:
            session.add(
                Memory(
                    game_id=game_id,
                    suspect_id=suspect_id,
                    content=content,
                    embedding=embedding,
                )
            )
            session.commit()

    def nearest(
        self, *, game_id: str, suspect_id: str, embedding: list[float], k: int
    ) -> list[MemoryHit]:
        distance = Memory.embedding.cosine_distance(embedding)
        statement = (
            select(Memory.content, distance.label("distance"))
            .where(Memory.game_id == game_id, Memory.suspect_id == suspect_id)
            .order_by(distance)
            .limit(k)
        )
        with self._session_factory() as session:
            rows = session.execute(statement).all()
        return [MemoryHit(content=row.content, score=float(row.distance)) for row in rows]


def remember(
    store: MemoryStore,
    *,
    game_id: str,
    suspect_id: str,
    content: str,
    llm: LLM | None = None,
) -> None:
    """Embed ``content`` and store it for this suspect."""
    client = llm or default_client()
    (embedding,) = client.embed([content])
    store.add(game_id=game_id, suspect_id=suspect_id, content=content, embedding=embedding)


def recall(
    store: MemoryStore,
    *,
    game_id: str,
    suspect_id: str,
    query: str,
    k: int = 5,
    llm: LLM | None = None,
) -> list[MemoryHit]:
    """Return the ``k`` memories most similar to ``query``."""
    client = llm or default_client()
    (embedding,) = client.embed([query])
    return store.nearest(game_id=game_id, suspect_id=suspect_id, embedding=embedding, k=k)
