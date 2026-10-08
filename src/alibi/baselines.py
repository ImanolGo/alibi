"""Baselines to compare the fine-tuned lie detector against.

Run these on the test split. The LLM judge is given the *same input* as the
model — one statement — and must guess whether it is a lie. Without the case
context this is an under-determined task (a claim like "the bag is his" is
neither true nor false in isolation), which is itself worth reporting: it sets a
realistic bar and shows how much the fine-tuned model has to learn from style
alone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .generator import render
from .llm import LLM, default_client, spend_today

Row = tuple[str, int]  # (statement, label) with 1 = a lie


class Verdict(BaseModel):
    verdict: Literal["TRUE", "FALSE"]


def f1_of_lies(true_labels: list[int], predicted: list[int]) -> float:
    """F1 for the positive class (1 = a lie)."""
    tp = sum(1 for t, p in zip(true_labels, predicted, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, predicted, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, predicted, strict=True) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def load_rows(path: Path | str) -> list[Row]:
    rows: list[Row] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                obj = json.loads(line)
                rows.append((obj["statement"], int(obj["label"])))
    if not rows:
        raise SystemExit(f"no rows in {path}; run `alibi export-dataset` first")
    return rows


def always_true(rows: list[Row]) -> list[int]:
    return [0] * len(rows)


def majority(rows: list[Row]) -> list[int]:
    labels = [label for _, label in rows]
    winner = max((0, 1), key=lambda label: labels.count(label))
    return [winner] * len(rows)


def llm_judge(rows: list[Row], *, llm: LLM | None = None) -> tuple[list[int], float, float]:
    """Predict with the judge model; return (predictions, latency_s, cost_usd)."""
    client = llm or default_client()
    start_cost = spend_today()
    started = time.perf_counter()
    predictions: list[int] = []
    for statement, _ in rows:
        messages = [
            {"role": "system", "content": render("lie_judge.jinja")},
            {"role": "user", "content": render("lie_judge_user.jinja", statement=statement)},
        ]
        verdict = client.structured(messages, Verdict, role="judge")
        predictions.append(1 if verdict.verdict == "FALSE" else 0)
    count = max(len(rows), 1)
    latency = (time.perf_counter() - started) / count
    cost = (spend_today() - start_cost) / count
    return predictions, latency, cost


@dataclass(frozen=True)
class BaselineReport:
    n: int
    always_true_f1: float
    majority_f1: float
    llm_judge_f1: float
    llm_judge_latency_s: float
    llm_judge_cost_usd: float


def evaluate_baselines(
    test_path: Path | str, *, llm: LLM | None = None, limit: int | None = None
) -> BaselineReport:
    rows = load_rows(test_path)
    if limit is not None:
        rows = rows[:limit]
    truths = [label for _, label in rows]
    llm_predictions, latency, cost = llm_judge(rows, llm=llm)
    return BaselineReport(
        n=len(rows),
        always_true_f1=f1_of_lies(truths, always_true(rows)),
        majority_f1=f1_of_lies(truths, majority(rows)),
        llm_judge_f1=f1_of_lies(truths, llm_predictions),
        llm_judge_latency_s=latency,
        llm_judge_cost_usd=cost,
    )
