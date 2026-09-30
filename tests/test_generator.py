from __future__ import annotations

from alibi.case import Case
from alibi.fake_llm import FakeLLM
from alibi.generator import generate_valid_case, render


def test_render_case_prompt_includes_setting_and_schema() -> None:
    text = render(
        "case.jinja",
        setting={"name": "Test Hall"},
        setting_yaml="name: Test Hall",
        schema="{}",
    )
    assert "Test Hall" in text
    assert "{}" in text


def test_valid_case_needs_no_repair(valid_case_dict: dict) -> None:
    fake = FakeLLM(structured=[valid_case_dict])
    case, violations = generate_valid_case({}, llm=fake)

    assert violations == []
    assert isinstance(case, Case)
    assert [call["method"] for call in fake.calls] == ["structured"]
    assert fake.calls[0]["role"] == "generator"


def test_invalid_case_triggers_one_repair(invalid_case_dict: dict, valid_case_dict: dict) -> None:
    fake = FakeLLM(structured=[invalid_case_dict, valid_case_dict])
    _, violations = generate_valid_case({}, llm=fake)

    assert violations == []
    assert len(fake.calls) == 2


def test_repairs_are_capped(invalid_case_dict: dict) -> None:
    fake = FakeLLM(structured=[invalid_case_dict] * 3)
    _, violations = generate_valid_case({}, llm=fake, max_repairs=2)

    assert violations  # still broken, reported honestly
    assert len(fake.calls) == 3  # initial + 2 repairs


def test_seed_is_forwarded(valid_case_dict: dict) -> None:
    fake = FakeLLM(structured=[valid_case_dict])
    generate_valid_case({}, llm=fake, seed=42)
    assert fake.calls[0]["kwargs"]["seed"] == 42
