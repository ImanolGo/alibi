"""Scripted LLM for tests. Never touches the network.

Usage::

    fake = FakeLLM(completions=["hello"], structured=[my_model_dict])
    fake.complete([{"role": "user", "content": "hi"}], role="generator")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class FakeLLM:
    """Deterministic stand-in for :class:`alibi.llm.LiteLLM`."""

    def __init__(
        self,
        *,
        completions: Sequence[str | Exception] | None = None,
        structured: Sequence[Any] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.completions = list(completions or [])
        self.structured_responses = list(structured or [])
        self.embedding_responses = list(embeddings or [])
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: Sequence[dict[str, str]], *, role: str, **kwargs: Any) -> str:
        self.calls.append(
            {"method": "complete", "role": role, "messages": list(messages), "kwargs": kwargs}
        )
        if not self.completions:
            raise AssertionError("FakeLLM.complete called with no scripted response left")
        item = self.completions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def structured(
        self,
        messages: Sequence[dict[str, str]],
        model_cls: type[ModelT],
        *,
        role: str,
        **kwargs: Any,
    ) -> ModelT:
        self.calls.append(
            {
                "method": "structured",
                "role": role,
                "model_cls": model_cls.__name__,
                "kwargs": kwargs,
            }
        )
        if not self.structured_responses:
            raise AssertionError("FakeLLM.structured called with no scripted response left")
        item = self.structured_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, model_cls):
            return item
        if isinstance(item, BaseModel):
            return model_cls.model_validate(item.model_dump())
        if isinstance(item, dict):
            return model_cls.model_validate(item)
        if isinstance(item, str):
            return model_cls.model_validate_json(item)
        raise TypeError(f"cannot turn {type(item)!r} into {model_cls.__name__}")

    def embed(self, texts: Sequence[str], *, role: str = "embedding") -> list[list[float]]:
        self.calls.append({"method": "embed", "role": role, "texts": list(texts)})
        if self.embedding_responses:
            return [list(vector) for vector in self.embedding_responses[: len(texts)]]
        return [[float(i)] for i, _ in enumerate(texts)]
