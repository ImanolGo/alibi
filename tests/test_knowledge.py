from __future__ import annotations

import pytest

from alibi.case import Case
from alibi.knowledge import knowledge_for, known_person_ids


def test_knowledge_contains_only_seen_events(valid_case: Case) -> None:
    knowledge = knowledge_for(valid_case, "butler_hobbs")
    event_ids = {event.id for event in knowledge.events}
    # Hobbs was in the kitchen at 21:00 and the library at 22:00...
    assert event_ids == {"event_supper", "event_library"}
    # ...and was never at drinks or in the conservatory.
    assert "event_drinks" not in event_ids
    assert "event_conservatory" not in event_ids


def test_witnesses_count_as_knowing(valid_case: Case) -> None:
    event_ids = {e.id for e in knowledge_for(valid_case, "dr_penrose").events}
    assert "event_drinks" in event_ids  # witness
    assert "event_kitchen_late" in event_ids  # actor


def test_knowledge_exposes_only_own_secrets(valid_case: Case) -> None:
    knowledge = knowledge_for(valid_case, "lady_blackwood")
    assert {s.id for s in knowledge.secrets} == {"secret_lover"}


def test_a_suspect_never_knows_the_murder_scene(valid_case: Case) -> None:
    # Dr Penrose was nowhere near the library at 22:00.
    event_ids = {e.id for e in knowledge_for(valid_case, "dr_penrose").events}
    assert "event_library" not in event_ids


def test_known_person_ids_are_shared_scenes(valid_case: Case) -> None:
    knowledge = knowledge_for(valid_case, "lady_blackwood")
    # Lady Blackwood saw Penrose at drinks; the butler was elsewhere.
    assert "dr_penrose" in known_person_ids(knowledge)
    assert "butler_hobbs" not in known_person_ids(knowledge)


def test_unknown_person_raises(valid_case: Case) -> None:
    with pytest.raises(KeyError):
        knowledge_for(valid_case, "mrs_white")
