from __future__ import annotations

from alibi.export import DatasetRow, rows_from_claims, split_by_case


def test_rows_from_claims_drops_unknown_and_blank() -> None:
    claims = [
        ("eval-000", "TRUE", "It was raining."),
        ("eval-000", "FALSE", "I was in the kitchen."),
        ("eval-001", "UNKNOWN", "Perhaps."),
        ("eval-001", "TRUE", "   "),
    ]
    rows = rows_from_claims(claims)
    assert [(row.statement, row.label, row.case) for row in rows] == [
        ("It was raining.", 0, "eval-000"),
        ("I was in the kitchen.", 1, "eval-000"),
    ]


def test_split_by_case_keeps_cases_on_one_side_only() -> None:
    rows = [DatasetRow(statement=f"s{i}", label=i % 2, case=f"case{i}") for i in range(10)]

    splits = split_by_case(rows, val_frac=0.2, test_frac=0.2, seed=1)

    train = {row.case for row in splits["train"]}
    val = {row.case for row in splits["val"]}
    test = {row.case for row in splits["test"]}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 10
    assert splits["val"] and splits["test"]


def test_every_case_is_placed() -> None:
    rows = [DatasetRow(statement="s", label=0, case=f"c{i}") for i in range(5)]
    splits = split_by_case(rows, seed=0)
    placed = {row.case for split in splits.values() for row in split}
    assert placed == {f"c{i}" for i in range(5)}
