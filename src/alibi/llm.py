"""The only module allowed to call an LLM.

Every model call in Alibi goes through here, with a ``role`` that maps to a
model in ``config/models.yaml``. Calls record tokens, cost and latency, emit a
trace span, and count against a daily spend cap.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time as _time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from .config import ROLE_NAMES, get_settings
from .tracing import span

log = logging.getLogger(__name__)

Message = dict[str, str]
ModelT = TypeVar("ModelT", bound=BaseModel)


class UnknownRoleError(ValueError):
    """Raised when a role is not declared in ``config/models.yaml``."""


class BudgetExceededError(RuntimeError):
    """Raised when a call would exceed ``ALIBI_DAILY_BUDGET_USD``."""


class StructuredOutputError(RuntimeError):
    """Raised when the model never returns valid structured output."""


@dataclass(frozen=True)
class CallUsage:
    """What one call cost."""

    role: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float


class _SpendLedger:
    """Thread-safe running total, reset at midnight (local time)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = date.today()
        self._total = 0.0

    def _roll(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self._total = 0.0

    def add(self, cost: float) -> None:
        with self._lock:
            self._roll()
            self._total += max(cost, 0.0)

    def total(self) -> float:
        with self._lock:
            self._roll()
            return self._total

    def reset(self) -> None:
        with self._lock:
            self._day = date.today()
            self._total = 0.0


_ledger = _SpendLedger()


def spend_today() -> float:
    """USD spent so far today across every role."""
    return _ledger.total()


def reset_spend() -> None:
    """Forget today's spend. Used by tests and a ``--reset-budget`` flag."""
    _ledger.reset()


def check_budget(extra: float = 0.0) -> None:
    """Raise :class:`BudgetExceededError` if ``extra`` would break the cap."""
    budget = get_settings().daily_budget_usd
    if budget < 0:
        return
    spent = _ledger.total()
    if spent + extra > budget:
        raise BudgetExceededError(
            f"daily budget of ${budget:.2f} reached (spent ${spent:.4f} today)"
        )


def resolve_model(role: str) -> str:
    """Map a role to a LiteLLM model string."""
    if role not in ROLE_NAMES:
        raise UnknownRoleError(f"unknown role {role!r}; known roles: {', '.join(ROLE_NAMES)}")
    model = get_settings().models.get(role)
    if not model:
        raise UnknownRoleError(f"no model configured for role {role!r}")
    return model


def _call_litellm(**kwargs: Any) -> Any:
    """The single network seam. Tests monkeypatch this."""
    import litellm

    return litellm.completion(**kwargs)


def _embed_litellm(**kwargs: Any) -> Any:
    import litellm

    return litellm.embedding(**kwargs)


def _cost_of(response: Any) -> float:
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:  # pricing unknown for some models
        return 0.0


def _tokens_of(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt, completion


def _text_of(response: Any) -> str:
    return response.choices[0].message.content or ""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _with_schema(messages: Sequence[Message], model_cls: type[BaseModel]) -> list[Message]:
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    instruction = (
        "Reply with a single JSON object and nothing else (no prose, no code "
        "fences). It must validate against this JSON Schema:\n" + schema
    )
    prepared = [dict(message) for message in messages]
    prepared.append({"role": "user", "content": instruction})
    return prepared


class LiteLLM:
    """Default LLM client. Wrap it only for tests (:mod:`alibi.fake_llm`)."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    # -- chat ---------------------------------------------------------------
    def complete_with_usage(
        self,
        messages: Sequence[Message],
        *,
        role: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        seed: int | None = None,
        **extra: Any,
    ) -> tuple[str, CallUsage]:
        model = resolve_model(role)
        check_budget()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        seed = self.seed if seed is None else seed
        if seed is not None:
            kwargs["seed"] = seed
        kwargs.update(extra)

        started = _time.perf_counter()
        with span("llm.complete", **{"llm.role": role, "llm.model": model}) as current:
            response = _call_litellm(**kwargs)
            latency = _time.perf_counter() - started
            prompt_tokens, completion_tokens = _tokens_of(response)
            cost = _cost_of(response)
            _ledger.add(cost)
            usage = CallUsage(
                role=role,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                latency_s=latency,
            )
            with contextlib.suppress(Exception):
                for key, value in (
                    ("llm.prompt_tokens", prompt_tokens),
                    ("llm.completion_tokens", completion_tokens),
                    ("llm.cost_usd", cost),
                    ("llm.latency_s", latency),
                ):
                    current.set_attribute(key, value)
        return _text_of(response), usage

    def complete(
        self,
        messages: Sequence[Message],
        *,
        role: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        seed: int | None = None,
        **extra: Any,
    ) -> str:
        text, _ = self.complete_with_usage(
            messages,
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            **extra,
        )
        return text

    # -- structured output --------------------------------------------------
    def structured(
        self,
        messages: Sequence[Message],
        model_cls: type[ModelT],
        *,
        role: str,
        temperature: float = 0.3,
        max_retries: int = 2,
        seed: int | None = None,
        **extra: Any,
    ) -> ModelT:
        """Return a validated ``model_cls``, retrying on bad output."""
        prepared = _with_schema(messages, model_cls)
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            text = self.complete(prepared, role=role, temperature=temperature, seed=seed, **extra)
            try:
                return model_cls.model_validate_json(_strip_code_fences(text))
            except (ValidationError, ValueError) as exc:
                last_error = exc
                log.warning(
                    "structured output invalid (attempt %d/%d) for %s: %s",
                    attempt + 1,
                    max_retries + 1,
                    model_cls.__name__,
                    exc,
                )
                if attempt < max_retries:
                    prepared = [*prepared, {"role": "assistant", "content": text}]
                    prepared.append(
                        {
                            "role": "user",
                            "content": (
                                f"That response was invalid: {exc}. "
                                "Return corrected JSON only, matching the schema."
                            ),
                        }
                    )
        raise StructuredOutputError(
            f"{model_cls.__name__} was not valid after {max_retries + 1} attempts: {last_error}"
        )

    # -- embeddings ---------------------------------------------------------
    def embed(self, texts: Sequence[str], *, role: str = "embedding") -> list[list[float]]:
        model = get_settings().embedding_model
        check_budget()
        started = _time.perf_counter()
        with span("llm.embed", **{"llm.role": role, "llm.model": model}):
            response = _embed_litellm(model=model, input=list(texts))
            latency = _time.perf_counter() - started
            cost = _cost_of(response)
            _ledger.add(cost)
            log.debug("embed %d texts in %.2fs for $%.5f", len(texts), latency, cost)
        return [item["embedding"] for item in response.data]


@runtime_checkable
class LLM(Protocol):
    """What higher layers (generator, suspect, solver) depend on."""

    def complete(self, messages: Sequence[Message], *, role: str, **kwargs: Any) -> str: ...

    def structured(
        self,
        messages: Sequence[Message],
        model_cls: type[ModelT],
        *,
        role: str,
        **kwargs: Any,
    ) -> ModelT: ...

    def embed(self, texts: Sequence[str], *, role: str = ...) -> list[list[float]]: ...


_default: LiteLLM | None = None


def default_client() -> LiteLLM:
    global _default
    if _default is None:
        _default = LiteLLM()
    return _default


def complete(messages: Sequence[Message], **kwargs: Any) -> str:
    return default_client().complete(messages, **kwargs)


def complete_with_usage(messages: Sequence[Message], **kwargs: Any) -> tuple[str, CallUsage]:
    return default_client().complete_with_usage(messages, **kwargs)


def structured[ModelT: BaseModel](
    messages: Sequence[Message], model_cls: type[ModelT], **kwargs: Any
) -> ModelT:
    return default_client().structured(messages, model_cls, **kwargs)


def embed(texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
    return default_client().embed(texts, **kwargs)
