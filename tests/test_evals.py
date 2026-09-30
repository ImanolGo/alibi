from __future__ import annotations

from alibi.evals import (
    GameRecord,
    case_valid_rate,
    check_gates,
    compute_metrics,
    contradiction_rate,
    cost_per_game,
    early_confession_rate,
    solve_rate,
)
from alibi.truth import StoredClaim


def claim(kind: str) -> StoredClaim:
    return StoredClaim(
        game_id="g",
        suspect_id="s",
        answer="a",
        claim="c",
        verdict="FALSE",
        kind=kind,  # type: ignore[arg-type]
        confessed=False,
    )


def record(
    *,
    valid: bool = True,
    solved: bool = False,
    outcome: str = "lost",
    claims: list[StoredClaim] | None = None,
    early_confession: bool = False,
    cost: float = 0.0,
) -> GameRecord:
    return GameRecord(
        game_id="g",
        case_id="c",
        valid=valid,
        outcome=outcome,
        solved=solved,
        claims=claims or [],
        early_confession=early_confession,
        cost_usd=cost,
    )


def test_case_valid_rate() -> None:
    assert case_valid_rate([record(valid=True), record(valid=False)]) == 0.5


def test_solve_rate() -> None:
    assert solve_rate([record(solved=True), record(solved=False), record(solved=True)]) == 2 / 3


def test_contradiction_rate_counts_only_contradictions_over_all_claims() -> None:
    claims = [claim("ok"), claim("intentional_lie"), claim("contradiction"), claim("contradiction")]
    assert contradiction_rate([record(claims=claims)]) == 0.5


def test_contradiction_rate_is_zero_without_claims() -> None:
    assert contradiction_rate([record()]) == 0.0


def test_early_confession_rate() -> None:
    assert early_confession_rate([record(early_confession=True), record()]) == 0.5


def test_cost_per_game() -> None:
    assert cost_per_game([record(cost=0.1), record(cost=0.3)]) == 0.2


def test_compute_metrics_has_every_metric() -> None:
    assert set(compute_metrics([])) == {
        "case_valid_rate",
        "solve_rate",
        "contradiction_rate",
        "early_confession_rate",
        "cost_per_game",
    }


def test_check_gates_reports_violations() -> None:
    metrics = {"solve_rate": 0.6, "contradiction_rate": 0.2}
    gates = {"solve_rate": {"min": 0.4, "max": 0.85}, "contradiction_rate": {"max": 0.08}}

    failures = check_gates(metrics, gates)

    assert len(failures) == 1
    assert failures[0].metric == "contradiction_rate"
    assert "<= 0.08" in str(failures[0])


def test_check_gates_passes_when_in_range() -> None:
    metrics = {"solve_rate": 0.6, "contradiction_rate": 0.03}
    gates = {"solve_rate": {"min": 0.4, "max": 0.85}, "contradiction_rate": {"max": 0.08}}
    assert check_gates(metrics, gates) == []
