"""Turn a short setting description into a validated :class:`~alibi.case.Case`.

One structured LLM call, then the deterministic validator; if it complains,
feed the violations back (at most ``max_repairs`` times). Creativity is the
model's job, correctness is the validator's.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .case import Case
from .config import get_settings, load_setting
from .llm import LLM, Message, default_client
from .validator import Violation, validate

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
MAX_REPAIRS = 2

_environment: Environment | None = None


def _env() -> Environment:
    global _environment
    if _environment is None:
        _environment = Environment(
            loader=FileSystemLoader(PROMPT_DIR),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
    return _environment


def render(template_name: str, **context: Any) -> str:
    """Render one of the ``prompts/*.jinja`` templates."""
    return _env().get_template(template_name).render(**context)


def _base_context(setting: dict[str, Any] | None) -> dict[str, Any]:
    resolved = load_setting() if setting is None else setting
    return {
        "setting": resolved,
        "setting_yaml": yaml.safe_dump(resolved, sort_keys=False).strip(),
        "schema": json.dumps(Case.model_json_schema(), indent=2),
    }


def _system_message() -> Message:
    return {"role": "system", "content": render("case_system.jinja")}


def generate_case(
    setting: dict[str, Any] | None = None,
    *,
    llm: LLM | None = None,
    seed: int | None = None,
) -> Case:
    """One structured call: setting description in, a whole case out."""
    client = llm or default_client()
    context = _base_context(setting)
    messages: Sequence[Message] = [
        _system_message(),
        {"role": "user", "content": render("case.jinja", **context)},
    ]
    return client.structured(
        messages, Case, role="generator", seed=seed, timeout=get_settings().generator_timeout_s
    )


def repair_case(
    case: Case,
    violations: list[Violation],
    setting: dict[str, Any] | None = None,
    *,
    llm: LLM | None = None,
    seed: int | None = None,
) -> Case:
    """Show the model its broken case plus the violations and ask for a fix."""
    client = llm or default_client()
    context = _base_context(setting)
    context["case_json"] = case.model_dump_json(indent=2)
    context["violations"] = [violation.model_dump() for violation in violations]
    messages: Sequence[Message] = [
        _system_message(),
        {"role": "user", "content": render("repair.jinja", **context)},
    ]
    return client.structured(
        messages, Case, role="generator", seed=seed, timeout=get_settings().generator_timeout_s
    )


def generate_valid_case(
    setting: dict[str, Any] | None = None,
    *,
    llm: LLM | None = None,
    seed: int | None = None,
    max_repairs: int = MAX_REPAIRS,
) -> tuple[Case, list[Violation]]:
    """Generate, validate, and repair until clean or repairs run out."""
    case = generate_case(setting, llm=llm, seed=seed)
    violations = validate(case)
    repairs = 0
    while violations and repairs < max_repairs:
        log.info(
            "repairing case %s: %d violation(s), attempt %d/%d",
            case.id,
            len(violations),
            repairs + 1,
            max_repairs,
        )
        case = repair_case(case, violations, setting, llm=llm, seed=seed)
        violations = validate(case)
        repairs += 1
    return case, violations
