from __future__ import annotations

import json
from pathlib import Path

from alibi.baselines import (
    Verdict,
    always_true,
    evaluate_baselines,
    f1_of_lies,
    llm_judge,
    load_rows,
    majority,
)
from alibi.fake_llm import FakeLLM


def write_test_file(tmp_path: Path, rows: list[tuple[str, int]]) -> Path:
    path = tmp_path / "test.jsonl"
    path.write_text(
        "\n".join(json.dumps({"statement": s, "label": label}) for s, label in rows),
        encoding="utf-8",
    )
    return path


def test_f1_of_lies() -> None:
    # tp=1, fp=1, fn=1 -> precision 0.5, recall 0.5 -> F1 0.5
    assert f1_of_lies([1, 1, 0, 0], [1, 0, 0, 1]) == 0.5
    assert f1_of_lies([1, 1], [1, 1]) == 1.0
    assert f1_of_lies([0, 0], [0, 0]) == 0.0


def test_load_rows(tmp_path: Path) -> None:
    path = write_test_file(tmp_path, [("a", 0), ("b", 1)])
    assert load_rows(path) == [("a", 0), ("b", 1)]


def test_trivial_baselines() -> None:
    rows = [("a", 1), ("b", 0), ("c", 1)]
    assert always_true(rows) == [0, 0, 0]
    assert majority(rows) == [1, 1, 1]  # two lies beat one truth


def test_llm_judge_predictions() -> None:
    rows = [("a", 0), ("b", 1)]
    fake = FakeLLM(structured=[Verdict(verdict="TRUE"), Verdict(verdict="FALSE")])

    predictions, _latency, _cost = llm_judge(rows, llm=fake)

    assert predictions == [0, 1]
    assert fake.calls[0]["role"] == "judge"


def test_evaluate_baselines_reports_everything(tmp_path: Path) -> None:
    rows = [("a", 1), ("b", 1), ("c", 1), ("d", 0)]
    path = write_test_file(tmp_path, rows)
    fake = FakeLLM(structured=[Verdict(verdict="FALSE")] * len(rows))

    report = evaluate_baselines(path, llm=fake)

    assert report.n == 4
    assert report.always_true_f1 == 0.0  # no true positives
    assert report.majority_f1 > 0.8  # predicts all lies: 3/3 recall, 1 false positive
    assert report.llm_judge_f1 == report.majority_f1  # the fake always says FALSE
