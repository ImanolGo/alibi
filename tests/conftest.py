"""Shared fixtures: a hand-made, validator-clean case (no LLM anywhere)."""

from __future__ import annotations

import pytest

from alibi import detector as detector_module
from alibi.case import Case
from alibi.config import Settings


@pytest.fixture(autouse=True)
def _no_detector_model(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Tests must not depend on a trained model being present on disk."""
    monkeypatch.setattr(
        detector_module,
        "get_settings",
        lambda: Settings(detector_dir=tmp_path / "no-model"),
    )


def base_case() -> dict:
    """A minimal sound case: butler_hobbs did it in the library at 22:00."""
    return {
        "id": "blackwood_manor_01",
        "title": "The Blackwood Cufflink",
        "setting": "Blackwood Manor, 1926, snowed in for the evening.",
        "time_of_death": "22:00",
        "crime_location": "library",
        "murderer": "butler_hobbs",
        "victim": "lord_blackwood",
        "persons": [
            {
                "id": "lord_blackwood",
                "name": "Lord Edmund Blackwood",
                "description": "Master of the house.",
                "is_victim": True,
                "secrets": [],
            },
            {
                "id": "butler_hobbs",
                "name": "Hobbs the butler",
                "description": "Impeccable, nervous, forever polishing something.",
                "secrets": [
                    {
                        "id": "secret_ledger",
                        "text": "has been skimming the household accounts for years",
                        "is_red_herring": False,
                    }
                ],
            },
            {
                "id": "lady_blackwood",
                "name": "Lady Beatrice Blackwood",
                "description": "The much younger widow-to-be, bored and brilliant.",
                "secrets": [
                    {
                        "id": "secret_lover",
                        "text": "was slipping out to meet a lover from the village",
                        "is_red_herring": True,
                    }
                ],
            },
            {
                "id": "dr_penrose",
                "name": "Dr Alistair Penrose",
                "description": "The family doctor, genial and perpetually short of money.",
                "secrets": [
                    {
                        "id": "secret_debts",
                        "text": "owes a great deal to a bookmaker in York",
                        "is_red_herring": False,
                    }
                ],
            },
        ],
        "locations": [
            {"id": "drawing_room", "name": "Drawing room", "description": ""},
            {"id": "library", "name": "Library", "description": "The crime scene."},
            {"id": "conservatory", "name": "Conservatory", "description": ""},
            {"id": "kitchen", "name": "Kitchen", "description": ""},
        ],
        "timeline": [
            {
                "id": "event_drinks",
                "time": "20:30",
                "location": "drawing_room",
                "actors": ["lady_blackwood"],
                "witnesses": ["dr_penrose"],
                "description": "Lady Blackwood pours drinks; Dr Penrose watches the clock.",
            },
            {
                "id": "event_supper",
                "time": "21:00",
                "location": "kitchen",
                "actors": ["butler_hobbs"],
                "witnesses": [],
                "description": "Hobbs arranges a late supper.",
            },
            {
                "id": "event_library",
                "time": "22:00",
                "location": "library",
                "actors": ["butler_hobbs", "lord_blackwood"],
                "witnesses": [],
                "description": "Hobbs brings brandy to the library.",
            },
            {
                "id": "event_conservatory",
                "time": "22:30",
                "location": "conservatory",
                "actors": ["lady_blackwood"],
                "witnesses": [],
                "description": "Lady Blackwood smokes alone among the ferns.",
            },
            {
                "id": "event_kitchen_late",
                "time": "22:30",
                "location": "kitchen",
                "actors": ["dr_penrose"],
                "witnesses": [],
                "description": "Penrose makes himself a nightcap.",
            },
        ],
        "clues": [
            {
                "id": "clue_ash",
                "location": "library",
                "description": "A smear of butler's ash on the brandy decanter.",
                "points_to": "butler_hobbs",
                "is_red_herring": False,
            },
            {
                "id": "clue_cufflink",
                "location": "library",
                "description": "A monogrammed cufflink torn from a servant's cuff.",
                "points_to": "butler_hobbs",
                "is_red_herring": False,
            },
            {
                "id": "clue_letter",
                "location": "drawing_room",
                "description": "A half-burned love letter in the grate.",
                "points_to": "lady_blackwood",
                "is_red_herring": True,
            },
            {
                "id": "clue_chip",
                "location": "conservatory",
                "description": "A betting chip from a York club.",
                "points_to": "dr_penrose",
                "is_red_herring": True,
            },
        ],
    }


@pytest.fixture
def valid_case_dict() -> dict:
    return base_case()


@pytest.fixture
def valid_case() -> Case:
    return Case.model_validate(base_case())


@pytest.fixture
def invalid_case_dict() -> dict:
    """Same case, but the murderer is absent at the time of death."""
    data = base_case()
    data["time_of_death"] = "23:00"
    return data
