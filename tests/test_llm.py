from __future__ import annotations

import types
from typing import Any

import pytest
from pydantic import BaseModel

from alibi import llm
from alibi.config import Settings
from alibi.llm import BudgetExceededError, StructuredOutputError, UnknownRoleError


class Widget(BaseModel):
    name: str
    count: int


def _response(content: str) -> Any:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
        usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=4),
    )


@pytest.fixture(autouse=True)
def _clean_ledger() -> None:
    llm.reset_spend()


def test_unknown_role_raises() -> None:
    with pytest.raises(UnknownRoleError):
        llm.resolve_model("nope")


def test_known_role_resolves_to_configured_model() -> None:
    assert llm.resolve_model("generator") == llm.get_settings().models["generator"]


def test_structured_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_call_litellm", lambda **_: _response('{"name": "a", "count": 1}'))
    monkeypatch.setattr(llm, "_cost_of", lambda _: 0.0)

    widget = llm.LiteLLM().structured([{"role": "user", "content": "hi"}], Widget, role="generator")
    assert widget == Widget(name="a", count=1)


def test_structured_retries_once_on_bad_output(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(["not json at all", '{"name": "fixed", "count": 2}'])
    calls: list[dict] = []

    def fake(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _response(next(responses))

    monkeypatch.setattr(llm, "_call_litellm", fake)
    monkeypatch.setattr(llm, "_cost_of", lambda _: 0.0)

    widget = llm.LiteLLM().structured([{"role": "user", "content": "hi"}], Widget, role="generator")
    assert widget.count == 2
    assert len(calls) == 2


def test_structured_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _response("still not json")

    monkeypatch.setattr(llm, "_call_litellm", fake)
    monkeypatch.setattr(llm, "_cost_of", lambda _: 0.0)

    with pytest.raises(StructuredOutputError):
        llm.LiteLLM().structured(
            [{"role": "user", "content": "hi"}], Widget, role="generator", max_retries=2
        )
    assert len(calls) == 3  # initial + 2 retries


def test_spend_cap_blocks_the_call_that_would_exceed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(daily_budget_usd=0.0, models={"generator": "openai/gpt-4o"})
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "_call_litellm", lambda **_: _response("hi"))
    monkeypatch.setattr(llm, "_cost_of", lambda _: 1.0)

    client = llm.LiteLLM()
    client.complete([{"role": "user", "content": "first"}], role="generator")  # spend is 0, allowed

    with pytest.raises(BudgetExceededError):
        client.complete([{"role": "user", "content": "second"}], role="generator")


def test_calls_carry_a_timeout_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _response("hi")

    monkeypatch.setattr(llm, "_call_litellm", fake)
    monkeypatch.setattr(llm, "_cost_of", lambda _: 0.0)

    llm.LiteLLM().complete([{"role": "user", "content": "a"}], role="generator")

    assert captured["timeout"] == llm.get_settings().request_timeout_s
    assert captured["num_retries"] == 2


def test_ledger_sums_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(daily_budget_usd=100.0, models={"generator": "openai/gpt-4o"})
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "_call_litellm", lambda **_: _response("hi"))
    monkeypatch.setattr(llm, "_cost_of", lambda _: 0.25)

    client = llm.LiteLLM()
    _, usage = client.complete_with_usage([{"role": "user", "content": "a"}], role="generator")
    assert usage.cost_usd == 0.25
    assert llm.spend_today() == pytest.approx(0.25)
