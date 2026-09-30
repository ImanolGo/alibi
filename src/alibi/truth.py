"""Truth labelling: turn a suspect's answer into claims labelled against the case.

The judge *does* see the ground truth (it is a fact-checker, not a detective).
A FALSE claim is an **intentional lie** when it is about the speaker's secret,
or about the murder when the speaker is the murderer; any other FALSE claim is a
**contradiction** — a bug in the prompt or agent worth counting.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .case import Case
from .generator import render
from .llm import LLM, default_client

Verdict = Literal["TRUE", "FALSE", "UNKNOWN"]
Kind = Literal["ok", "unknown", "intentional_lie", "contradiction"]


class ClaimLabel(BaseModel):
    """One atomic claim and its verdict against the ground truth."""

    claim: str
    verdict: Verdict = "UNKNOWN"
    about_secret: bool = False
    about_murder: bool = False
    note: str = ""


class AnswerJudgement(BaseModel):
    """The labelled claims in one suspect answer, plus whether they confessed."""

    claims: list[ClaimLabel] = Field(default_factory=list)
    confessed: bool = False


def claim_kind(label: ClaimLabel, *, is_murderer: bool) -> Kind:
    """Classify a labelled claim: expected lie, contradiction, or fine.

    Pure and deterministic — this is the part we unit-test.
    """
    if label.verdict == "TRUE":
        return "ok"
    if label.verdict == "UNKNOWN":
        return "unknown"
    # FALSE from here on.
    if label.about_secret or (is_murderer and label.about_murder):
        return "intentional_lie"
    return "contradiction"


def _ground_truth_context(case: Case, suspect_id: str) -> dict[str, Any]:
    person = case.person(suspect_id)
    victim = case.victim_person()
    murderer = case.person(case.murderer)

    def name_of(person_id: str) -> str:
        found = case.person(person_id)
        return found.name if found else person_id

    return {
        "suspect_id": suspect_id,
        "suspect_name": person.name if person else suspect_id,
        "is_murderer": case.murderer == suspect_id,
        "murderer_id": case.murderer,
        "murderer_name": murderer.name if murderer else case.murderer,
        "victim_name": victim.name if victim else case.victim,
        "time_of_death": case.time_of_death,
        "crime_location": case.crime_location,
        "secrets": [secret.text for secret in (person.secrets if person else [])],
        "timeline": [
            {
                "time": event.time,
                "location": event.location,
                "description": event.description,
                "actors": [name_of(pid) for pid in event.actors],
            }
            for event in case.timeline
        ],
        "clues": [
            {"id": clue.id, "description": clue.description, "points_to": clue.points_to}
            for clue in case.clues
        ],
    }


def judge_answer(
    case: Case,
    suspect_id: str,
    answer: str,
    *,
    llm: LLM | None = None,
) -> AnswerJudgement:
    """Ask the judge model to label every claim in ``answer``."""
    client = llm or default_client()
    context = _ground_truth_context(case, suspect_id)
    context["answer"] = answer
    messages = [
        {"role": "system", "content": render("truth_judge_system.jinja")},
        {"role": "user", "content": render("truth_judge.jinja", **context)},
    ]
    return client.structured(messages, AnswerJudgement, role="judge")


def kinds_for(judgement: AnswerJudgement, *, is_murderer: bool) -> list[Kind]:
    return [claim_kind(label, is_murderer=is_murderer) for label in judgement.claims]


class StoredClaim(BaseModel):
    """A claim flattened for storage/export (M5 dataset)."""

    game_id: str
    suspect_id: str
    answer: str
    claim: str
    verdict: Verdict
    kind: Kind
    confessed: bool
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def flatten(
    game_id: str,
    suspect_id: str,
    answer: str,
    judgement: AnswerJudgement,
    *,
    is_murderer: bool,
) -> list[StoredClaim]:
    return [
        StoredClaim(
            game_id=game_id,
            suspect_id=suspect_id,
            answer=answer,
            claim=label.claim,
            verdict=label.verdict,
            kind=claim_kind(label, is_murderer=is_murderer),
            confessed=judgement.confessed,
        )
        for label in judgement.claims
    ]
