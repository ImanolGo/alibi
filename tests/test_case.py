from __future__ import annotations

import pytest

from alibi.case import Case, format_time, parse_time


def test_parse_and_format_time_round_trip() -> None:
    assert parse_time("00:00") == 0
    assert parse_time("22:00") == 22 * 60
    assert format_time(parse_time("07:05")) == "07:05"


@pytest.mark.parametrize("bad", ["24:00", "9:00", "22:60", "2200", "quarter past ten"])
def test_parse_time_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_time(bad)


def test_bad_time_fails_validation(valid_case_dict: dict) -> None:
    valid_case_dict["time_of_death"] = "25:99"
    with pytest.raises(ValueError):
        Case.model_validate(valid_case_dict)


def test_lookups_and_derived_views(valid_case: Case) -> None:
    assert valid_case.person("butler_hobbs") is not None
    assert valid_case.person("nobody") is None
    assert valid_case.location("library").name == "Library"  # type: ignore[union-attr]
    assert valid_case.clue("clue_ash") is not None

    suspect_ids = {p.id for p in valid_case.suspects()}
    assert suspect_ids == {"butler_hobbs", "lady_blackwood", "dr_penrose"}
    assert valid_case.victim_person().id == "lord_blackwood"  # type: ignore[union-attr]
