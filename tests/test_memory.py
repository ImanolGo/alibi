from __future__ import annotations

from alibi.fake_llm import FakeLLM
from alibi.memory import MemoryHit, recall, remember


class FakeStore:
    def __init__(self, hits: list[MemoryHit] | None = None) -> None:
        self.added: list[dict] = []
        self.queries: list[dict] = []
        self.hits = hits or []

    def add(self, *, game_id: str, suspect_id: str, content: str, embedding: list[float]) -> None:
        self.added.append(
            {
                "game_id": game_id,
                "suspect_id": suspect_id,
                "content": content,
                "embedding": embedding,
            }
        )

    def nearest(
        self, *, game_id: str, suspect_id: str, embedding: list[float], k: int
    ) -> list[MemoryHit]:
        self.queries.append(
            {"game_id": game_id, "suspect_id": suspect_id, "embedding": embedding, "k": k}
        )
        return self.hits


def test_remember_embeds_and_stores() -> None:
    store = FakeStore()
    llm = FakeLLM(embeddings=[[0.1, 0.2, 0.3]])

    remember(store, game_id="g1", suspect_id="hobbs", content="I was in the kitchen", llm=llm)

    assert llm.calls[0]["method"] == "embed"
    assert llm.calls[0]["texts"] == ["I was in the kitchen"]
    assert store.added == [
        {
            "game_id": "g1",
            "suspect_id": "hobbs",
            "content": "I was in the kitchen",
            "embedding": [0.1, 0.2, 0.3],
        }
    ]


def test_recall_embeds_query_and_forwards_k() -> None:
    store = FakeStore(hits=[MemoryHit(content="earlier", score=0.12)])
    llm = FakeLLM(embeddings=[[1.0, 0.0]])

    hits = recall(store, game_id="g1", suspect_id="hobbs", query="where were you?", k=3, llm=llm)

    assert [hit.content for hit in hits] == ["earlier"]
    assert llm.calls[0]["texts"] == ["where were you?"]
    assert store.queries == [
        {"game_id": "g1", "suspect_id": "hobbs", "embedding": [1.0, 0.0], "k": 3}
    ]
