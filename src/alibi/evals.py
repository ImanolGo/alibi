"""Run games, measure them, and gate changes on the numbers.

``alibi eval --games N`` generates cases, lets the solver play them, labels every
suspect answer with :mod:`alibi.truth`, and computes the metrics below. Each
metric is a pure function of the per-game records, so it is unit-tested on fake
transcripts without touching a model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .case import Case
from .config import get_settings
from .game import Game, InMemoryGameStore
from .generator import generate_valid_case
from .llm import LLM, BudgetExceededError, default_client, spend_today
from .memory import MemoryStore
from .solver import MAX_STEPS, SolverAgent
from .suspect import SuspectTeam
from .truth import StoredClaim, flatten, judge_answer

log = logging.getLogger(__name__)


@dataclass
class GameRecord:
    game_id: str
    case_id: str
    valid: bool
    outcome: str
    solved: bool
    claims: list[StoredClaim] = field(default_factory=list)
    early_confession: bool = False
    cost_usd: float = 0.0
    steps: int = 0


# --- metrics (pure, unit-tested) -------------------------------------------
def _mean(flags: list[bool]) -> float:
    return sum(1 for flag in flags if flag) / len(flags) if flags else 0.0


def case_valid_rate(records: list[GameRecord]) -> float:
    """Fraction of generated cases that passed the deterministic validator."""
    return _mean([record.valid for record in records])


def solve_rate(records: list[GameRecord]) -> float:
    """Fraction of games the solver won (named the murderer). Too high = too easy."""
    return _mean([record.solved for record in records])


def contradiction_rate(records: list[GameRecord]) -> float:
    """Unintended false claims (bugs) divided by all claims made."""
    total = sum(len(record.claims) for record in records)
    contradictions = sum(
        1 for record in records for claim in record.claims if claim.kind == "contradiction"
    )
    return contradictions / total if total else 0.0


def early_confession_rate(records: list[GameRecord]) -> float:
    """Fraction of games where the murderer confessed before 2 real clues were shown."""
    return _mean([record.early_confession for record in records])


def cost_per_game(records: list[GameRecord]) -> float:
    """Average USD spent per game."""
    return sum(record.cost_usd for record in records) / len(records) if records else 0.0


METRICS: dict[str, Callable[[list[GameRecord]], float]] = {
    "case_valid_rate": case_valid_rate,
    "solve_rate": solve_rate,
    "contradiction_rate": contradiction_rate,
    "early_confession_rate": early_confession_rate,
    "cost_per_game": cost_per_game,
}


def compute_metrics(records: list[GameRecord]) -> dict[str, float]:
    return {name: function(records) for name, function in METRICS.items()}


# --- gates -----------------------------------------------------------------
@dataclass
class GateFailure:
    metric: str
    value: float
    rule: str

    def __str__(self) -> str:
        return f"{self.metric}={self.value:.3f} violates {self.rule}"


def load_gates(path: Path | None = None) -> dict[str, dict[str, float]]:
    path = path or (get_settings().project_root / "config" / "gates.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_gates(metrics: dict[str, float], gates: dict[str, dict[str, float]]) -> list[GateFailure]:
    failures: list[GateFailure] = []
    for metric, rule in gates.items():
        value = metrics.get(metric)
        if value is None:
            continue
        if "min" in rule and value < rule["min"]:
            failures.append(GateFailure(metric, value, f">= {rule['min']}"))
        if "max" in rule and value > rule["max"]:
            failures.append(GateFailure(metric, value, f"<= {rule['max']}"))
    return failures


# --- recording responder ---------------------------------------------------
class RecordingResponder:
    """Wraps a SuspectTeam and remembers the clues shown at each answer."""

    def __init__(self, inner: SuspectTeam) -> None:
        self.inner = inner
        self._clues: dict[str, list[str]] = {}
        self.answers: list[tuple[str, str, str, list[str]]] = []

    def set_clues(self, suspect_id: str, clue_ids: list[str]) -> None:
        self._clues[suspect_id] = list(clue_ids)
        self.inner.set_clues(suspect_id, clue_ids)

    def answer(self, suspect_id: str, question: str) -> str:
        answer = self.inner.answer(suspect_id, question)
        self.answers.append((suspect_id, question, answer, list(self._clues.get(suspect_id, []))))
        return answer


def _pointing_clues(case: Case, suspect_id: str, clue_ids: list[str]) -> int:
    count = 0
    for clue_id in clue_ids:
        clue = case.clue(clue_id)
        if clue and not clue.is_red_herring and clue.points_to == suspect_id:
            count += 1
    return count


def run_eval(
    games: int,
    *,
    seed: int | None = None,
    setting: dict[str, Any] | None = None,
    llm: LLM | None = None,
    memory_store: MemoryStore | None = None,
    max_steps: int = MAX_STEPS,
) -> list[GameRecord]:
    """Generate ``games`` cases, play each with the solver, label, and record."""
    client = llm or default_client()
    records: list[GameRecord] = []
    for index in range(games):
        before = spend_today()
        game_id = f"eval-{index:03d}"
        log.info("eval game %d/%d: generating a case", index + 1, games)
        try:
            case, violations = generate_valid_case(
                setting, llm=client, seed=None if seed is None else seed + index
            )
        except BudgetExceededError:
            break
        except Exception as exc:  # noqa: BLE001 - one bad game must not kill the run
            log.warning("eval game %d: generation failed: %s", index + 1, exc)
            records.append(
                GameRecord(
                    game_id=game_id,
                    case_id="",
                    valid=False,
                    outcome="error",
                    solved=False,
                    cost_usd=spend_today() - before,
                )
            )
            continue

        responder = RecordingResponder(
            SuspectTeam(case, game_id=game_id, memory_store=memory_store, llm=client)
        )
        game = Game(case, game_id=game_id, store=InMemoryGameStore(), responder=responder)
        solver = SolverAgent(game, llm=client, max_steps=max_steps)
        log.info("eval game %d/%d: solver is playing %s", index + 1, games, case.id)
        try:
            solver.play()
        except BudgetExceededError:
            break
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "eval game %d: solver failed after %d steps: %s",
                index + 1,
                len(solver.actions),
                exc,
            )
            records.append(
                GameRecord(
                    game_id=game_id,
                    case_id=case.id,
                    valid=not violations,
                    outcome="error",
                    solved=False,
                    cost_usd=spend_today() - before,
                    steps=len(solver.actions),
                )
            )
            continue
        log.info(
            "eval game %d/%d: %s in %d steps",
            index + 1,
            games,
            game.state.outcome,
            len(solver.actions),
        )

        claims: list[StoredClaim] = []
        early_confession = False
        for suspect_id, _question, answer, clue_ids in responder.answers:
            is_murderer = case.murderer == suspect_id
            try:
                judgement = judge_answer(case, suspect_id, answer, llm=client)
            except BudgetExceededError:
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("eval game %d: labelling failed: %s", index + 1, exc)
                continue
            claims.extend(flatten(game_id, suspect_id, answer, judgement, is_murderer=is_murderer))
            if (
                is_murderer
                and judgement.confessed
                and _pointing_clues(case, suspect_id, clue_ids) < 2
            ):
                early_confession = True
        log.info("eval game %d/%d: labelled %d claims", index + 1, games, len(claims))

        records.append(
            GameRecord(
                game_id=game_id,
                case_id=case.id,
                valid=not violations,
                outcome=game.state.outcome,
                solved=game.state.outcome == "won",
                claims=claims,
                early_confession=early_confession,
                cost_usd=spend_today() - before,
                steps=len(solver.actions),
            )
        )
    return records


# --- reporting -------------------------------------------------------------
def render_report(
    records: list[GameRecord], metrics: dict[str, float], failures: list[GateFailure]
) -> str:
    lines = [
        f"# Eval results — {date.today().isoformat()}",
        "",
        f"- Games run: {len(records)}",
        "- Outcomes: " + ", ".join(record.outcome for record in records),
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.3f} |" for name, value in metrics.items()]
    total_claims = sum(len(record.claims) for record in records)
    lines += ["", f"Claims labelled: {total_claims}", ""]
    if failures:
        lines += ["## Gate failures", ""]
        lines += [f"- {failure}" for failure in failures]
    else:
        lines += ["All gates passed."]
    return "\n".join(lines) + "\n"


def write_report(content: str, path: Path | None = None) -> Path:
    path = path or (
        get_settings().project_root / "docs" / "results" / f"eval-{date.today().isoformat()}.md"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_claims(records: list[GameRecord], session_factory=None) -> int:
    """Persist labelled claims to Postgres (best effort; used by M5 exports)."""
    from .db import Claim, get_engine, get_session_factory, init_db

    init_db(get_engine())
    session_factory = session_factory or get_session_factory()
    saved = 0
    with session_factory() as session:
        for record in records:
            for claim in record.claims:
                session.add(
                    Claim(
                        game_id=claim.game_id,
                        suspect_id=claim.suspect_id,
                        answer=claim.answer,
                        claim=claim.claim,
                        verdict=claim.verdict,
                        kind=claim.kind,
                        confessed=claim.confessed,
                    )
                )
                saved += 1
        session.commit()
    return saved
