"""Export labelled claims as a fine-tuning dataset, split by case.

Every eval game generates its own case, so splitting by ``game_id`` is splitting
by case: no case appears in two splits, so there is no leakage between train and
test. Only TRUE/FALSE claims are exported; UNKNOWN ones are dropped.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from .db import Claim, get_session_factory


@dataclass(frozen=True)
class DatasetRow:
    """One training example: a statement and whether it is a lie."""

    statement: str
    label: int  # 1 = FALSE (a lie), 0 = TRUE
    case: str


def rows_from_claims(claims: list[tuple[str, str, str]]) -> list[DatasetRow]:
    """Turn ``(game_id, verdict, statement)`` rows into labelled examples."""
    rows: list[DatasetRow] = []
    for game_id, verdict, statement in claims:
        if verdict not in ("TRUE", "FALSE"):
            continue
        if not statement.strip():
            continue
        rows.append(
            DatasetRow(statement=statement, label=1 if verdict == "FALSE" else 0, case=game_id)
        )
    return rows


def split_by_case(
    rows: list[DatasetRow],
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[DatasetRow]]:
    """Split rows into train/val/test with whole cases on one side only."""
    cases = sorted({row.case for row in rows})
    random.Random(seed).shuffle(cases)

    n = len(cases)
    n_test = max(1, int(n * test_frac)) if n >= 3 else 0
    n_val = max(1, int(n * val_frac)) if n >= 3 else 0
    test_cases = set(cases[:n_test])
    val_cases = set(cases[n_test : n_test + n_val])

    splits: dict[str, list[DatasetRow]] = {"train": [], "val": [], "test": []}
    for row in rows:
        if row.case in test_cases:
            splits["test"].append(row)
        elif row.case in val_cases:
            splits["val"].append(row)
        else:
            splits["train"].append(row)
    return splits


def load_claims(session_factory=None) -> list[tuple[str, str, str]]:
    """Read ``(game_id, verdict, statement)`` for every stored claim."""
    session_factory = session_factory or get_session_factory()
    with session_factory() as session:
        result = session.execute(select(Claim.game_id, Claim.verdict, Claim.claim)).all()
    return [(row[0], row[1], row[2]) for row in result]


def export_dataset(
    out_dir: Path | str,
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    session_factory=None,
) -> dict[str, int]:
    """Write ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` and return counts."""
    rows = rows_from_claims(load_claims(session_factory))
    splits = split_by_case(rows, val_frac=val_frac, test_frac=test_frac, seed=seed)

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, split_rows in splits.items():
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps({"statement": row.statement, "label": row.label}) + "\n")
        counts[name] = len(split_rows)
    return counts
