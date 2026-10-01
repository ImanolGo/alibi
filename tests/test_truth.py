from __future__ import annotations

from alibi.case import Case
from alibi.fake_llm import FakeLLM
from alibi.truth import (
    AnswerJudgement,
    ClaimLabel,
    claim_kind,
    flatten,
    judge_answer,
    kinds_for,
)


def label(verdict: str, *, secret: bool = False, murder: bool = False) -> ClaimLabel:
    return ClaimLabel(claim="x", verdict=verdict, about_secret=secret, about_murder=murder)


# --- deterministic classification -----------------------------------------
def test_true_claim_is_ok() -> None:
    assert claim_kind(label("TRUE"), is_murderer=False) == "ok"


def test_unknown_claim_is_unknown() -> None:
    assert claim_kind(label("UNKNOWN"), is_murderer=True) == "unknown"


def test_false_about_secret_is_an_intentional_lie() -> None:
    assert claim_kind(label("FALSE", secret=True), is_murderer=False) == "intentional_lie"


def test_false_about_the_murder_is_a_lie_for_the_murderer() -> None:
    assert claim_kind(label("FALSE", murder=True), is_murderer=True) == "intentional_lie"


def test_false_about_the_murder_is_a_contradiction_for_an_innocent() -> None:
    assert claim_kind(label("FALSE", murder=True), is_murderer=False) == "contradiction"


def test_false_about_nothing_relevant_is_a_contradiction() -> None:
    assert claim_kind(label("FALSE"), is_murderer=True) == "contradiction"


def test_kinds_for_maps_every_claim() -> None:
    judgement = AnswerJudgement(claims=[label("TRUE"), label("FALSE", secret=True), label("FALSE")])
    assert kinds_for(judgement, is_murderer=False) == ["ok", "intentional_lie", "contradiction"]


# --- flattening for storage ------------------------------------------------
def test_flatten_carries_the_verdict_and_kind() -> None:
    judgement = AnswerJudgement(claims=[label("TRUE"), label("FALSE", murder=True)], confessed=True)
    stored = flatten("g1", "butler_hobbs", "I was in the kitchen.", judgement, is_murderer=True)
    assert [row.kind for row in stored] == ["ok", "intentional_lie"]
    assert all(row.confessed for row in stored)
    assert stored[0].answer == "I was in the kitchen."


# --- the judge call --------------------------------------------------------
def test_judge_answer_uses_the_judge_role_and_sees_ground_truth(valid_case: Case) -> None:
    reply = AnswerJudgement(claims=[label("FALSE", secret=True)])
    fake = FakeLLM(structured=[reply])
    judgement = judge_answer(valid_case, "butler_hobbs", "I never touched the books.", llm=fake)

    assert judgement.claims[0].verdict == "FALSE"
    assert fake.calls[0]["role"] == "judge"
    assert fake.calls[0]["model_cls"] == "AnswerJudgement"
    # The judge prompt includes the ground truth (murderer id) so it can judge.
    # (The solver, by contrast, never sees this — see tests/test_solver.py.)
