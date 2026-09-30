"""What a suspect knows: pure functions, no LLM.

Information hiding lives here. A suspect's prompt may contain only its persona,
its own secrets, and the timeline events it took part in or witnessed — never
the whole case, never another suspect's secret.
"""

from __future__ import annotations

from dataclasses import dataclass

from .case import Case, Person, Secret, TimelineEvent


@dataclass(frozen=True)
class Knowledge:
    """The slice of a case a single person may legitimately know."""

    person: Person
    events: list[TimelineEvent]
    secrets: list[Secret]


def knowledge_for(case: Case, person_id: str) -> Knowledge:
    """Events the person took part in or witnessed, plus their own secrets."""
    person = case.person(person_id)
    if person is None:
        raise KeyError(f"unknown person {person_id!r}")

    events = sorted(
        (
            event
            for event in case.timeline
            if person_id in event.actors or person_id in event.witnesses
        ),
        key=lambda event: event.minutes,
    )
    return Knowledge(person=person, events=events, secrets=list(person.secrets))


def known_person_ids(knowledge: Knowledge) -> set[str]:
    """People the person has actually shared a scene with."""
    ids: set[str] = set()
    for event in knowledge.events:
        ids.update(event.actors)
        ids.update(event.witnesses)
    ids.discard(knowledge.person.id)
    return ids
