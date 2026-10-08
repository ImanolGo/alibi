"""One passing and one failing test per validator check."""

from __future__ import annotations

from alibi.case import Case
from alibi.validator import validate


def codes(case: Case) -> set[str]:
    return {violation.code for violation in validate(case)}


# --- the clean case --------------------------------------------------------
def test_valid_case_has_no_violations(valid_case: Case) -> None:
    assert validate(valid_case) == []


# --- check 1: exactly one murderer, and a suspect --------------------------
def test_check1_pass(valid_case: Case) -> None:
    assert "murderer_is_victim" not in codes(valid_case)


def test_check1_fail_murderer_is_victim(valid_case: Case) -> None:
    broken = valid_case.model_copy(deep=True)
    broken.murderer = "lord_blackwood"
    assert "murderer_is_victim" in codes(broken)


# --- check 2: murderer at the scene at the time of death -------------------
def test_check2_pass(valid_case: Case) -> None:
    assert "murderer_not_at_scene" not in codes(valid_case)


def test_check2_fail_murderer_absent(valid_case: Case) -> None:
    broken = valid_case.model_copy(deep=True)
    broken.time_of_death = "23:00"
    assert "murderer_not_at_scene" in codes(broken)


# --- check 3: nobody in two places at once ---------------------------------
def test_check3_pass(valid_case: Case) -> None:
    assert "double_booked" not in codes(valid_case)


def test_check3_fail_double_booked(valid_case: Case) -> None:
    broken = valid_case.model_copy(deep=True)
    broken.timeline.append(
        broken.timeline[0].model_copy(
            update={
                "id": "event_clash",
                "time": "22:00",
                "location": "drawing_room",
                "actors": ["butler_hobbs"],
                "witnesses": [],
            }
        )
    )
    assert "double_booked" in codes(broken)


# --- check 4: secrets ------------------------------------------------------
def test_check4_pass(valid_case: Case) -> None:
    assert "suspect_without_secret" not in codes(valid_case)
    assert "no_red_herring_secret" not in codes(valid_case)


def test_check4_fail_suspect_without_secret(valid_case: Case) -> None:
    broken = valid_case.model_copy(deep=True)
    broken.person("dr_penrose").secrets = []  # type: ignore[union-attr]
    assert "suspect_without_secret" in codes(broken)


def test_check4_fail_no_red_herring(valid_case: Case) -> None:
    broken = valid_case.model_copy(deep=True)
    for person in broken.persons:
        for secret in person.secrets:
            secret.is_red_herring = False
    assert "no_red_herring_secret" in codes(broken)


# --- check 5: at least two real clues point to the murderer ----------------
def test_check5_pass(valid_case: Case) -> None:
    assert "insufficient_clues" not in codes(valid_case)


def test_check5_fail_only_one_clue(valid_case: Case) -> None:
    broken = valid_case.model_copy(deep=True)
    broken.clue("clue_cufflink").is_red_herring = True  # type: ignore[union-attr]
    assert "insufficient_clues" in codes(broken)


# --- extra check: innocents must be equally implicated (difficulty) --------
def test_spread_pass(valid_case: Case) -> None:
    assert "insufficient_spread" not in codes(valid_case)


def test_spread_fail_when_only_one_innocent_implicated(valid_case: Case) -> None:
    broken = valid_case.model_copy(deep=True)
    # Turn the doctor's red herring into a real clue: only Lady Blackwood's
    # red herring remains, so one innocent is implicated instead of two.
    broken.clue("clue_chip").is_red_herring = False  # type: ignore[union-attr]
    assert "insufficient_spread" in codes(broken)


# --- check 6: every referenced id exists -----------------------------------
def test_check6_pass(valid_case: Case) -> None:
    assert "unknown_reference" not in codes(valid_case)


def test_check6_fail_unknown_actor(valid_case: Case) -> None:
    broken = valid_case.model_copy(deep=True)
    broken.timeline[1].actors = ["ghost_of_christmas_past"]
    assert "unknown_reference" in codes(broken)


def test_check6_fail_unknown_location(valid_case: Case) -> None:
    broken = valid_case.model_copy(deep=True)
    broken.crime_location = "the_moon"
    assert "unknown_reference" in codes(broken)
