"""Deterministic checks on a generated case.

The LLM is creative; this module is the correctness gate. ``validate`` returns
an empty list for a good case, otherwise one :class:`Violation` per problem —
those messages are fed back to the LLM by the generator's repair loop.
"""

from __future__ import annotations

from pydantic import BaseModel

from .case import Case, format_time, parse_time


class Violation(BaseModel):
    code: str
    message: str


# --- check 1: exactly one murderer, and a suspect --------------------------
def _check_murderer(case: Case) -> list[Violation]:
    out: list[Violation] = []
    murderer = case.person(case.murderer)
    if murderer is None:
        return out  # unknown id is reported by _check_references
    if murderer.is_victim:
        out.append(
            Violation(
                code="murderer_is_victim",
                message=f"murderer {case.murderer!r} is marked as the victim",
            )
        )
    return out


# --- check 2: murderer at the scene, at the time of death ------------------
def _check_murderer_at_scene(case: Case) -> list[Violation]:
    death = parse_time(case.time_of_death)
    at_scene = any(
        case.murderer in event.actors
        and event.location == case.crime_location
        and event.minutes == death
        for event in case.timeline
    )
    if at_scene:
        return []
    return [
        Violation(
            code="murderer_not_at_scene",
            message=(
                f"murderer {case.murderer!r} is not an actor at "
                f"{case.crime_location!r} at {case.time_of_death} (time of death)"
            ),
        )
    ]


# --- check 3: nobody is in two places at once ------------------------------
def _check_double_booked(case: Case) -> list[Violation]:
    places: dict[int, dict[str, set[str]]] = {}
    for event in case.timeline:
        for person_id in set(event.actors) | set(event.witnesses):
            by_person = places.setdefault(event.minutes, {})
            by_person.setdefault(person_id, set()).add(event.location)

    out: list[Violation] = []
    for minutes, by_person in sorted(places.items()):
        for person_id, locations in sorted(by_person.items()):
            if len(locations) > 1:
                out.append(
                    Violation(
                        code="double_booked",
                        message=(
                            f"{person_id!r} is in {sorted(locations)} at {format_time(minutes)}"
                        ),
                    )
                )
    return out


# --- check 4: secrets ------------------------------------------------------
def _check_secrets(case: Case) -> list[Violation]:
    out: list[Violation] = []
    for suspect in case.suspects():
        if not suspect.secrets:
            out.append(
                Violation(
                    code="suspect_without_secret",
                    message=f"suspect {suspect.id!r} has no secret",
                )
            )

    innocents = [p for p in case.suspects() if p.id != case.murderer]
    has_red_herring = any(
        secret.is_red_herring for person in innocents for secret in person.secrets
    )
    if not has_red_herring:
        out.append(
            Violation(
                code="no_red_herring_secret",
                message="no innocent suspect has a red-herring secret to look suspicious",
            )
        )
    return out


# --- check 5: enough real evidence against the murderer --------------------
def _check_pointing_clues(case: Case) -> list[Violation]:
    pointing = [
        clue for clue in case.clues if not clue.is_red_herring and clue.points_to == case.murderer
    ]
    if len(pointing) < 2:
        return [
            Violation(
                code="insufficient_clues",
                message=(
                    f"only {len(pointing)} non-red-herring clue(s) point to the murderer; "
                    "at least 2 are required"
                ),
            )
        ]
    return []


# --- extra check (difficulty): innocents must be equally implicated ---------
def _check_suspicion_spread(case: Case) -> list[Violation]:
    red_herrings = [clue for clue in case.clues if clue.is_red_herring]
    innocents_implicated = {
        clue.points_to
        for clue in red_herrings
        if clue.points_to is not None and clue.points_to != case.murderer
    }
    if len(red_herrings) < 2 or len(innocents_implicated) < 2:
        return [
            Violation(
                code="insufficient_spread",
                message=(
                    "at least two red-herring clues must point at two different innocents, "
                    "so the culprit is not the only suspect a clue implicates"
                ),
            )
        ]
    return []


# --- check 6: every referenced id exists -----------------------------------
def _check_references(case: Case) -> list[Violation]:
    out: list[Violation] = []
    person_ids = {p.id for p in case.persons}
    location_ids = {loc.id for loc in case.locations}

    if case.murderer not in person_ids:
        out.append(
            Violation(
                code="unknown_reference",
                message=f"murderer {case.murderer!r} is not a person",
            )
        )
    if case.victim not in person_ids:
        out.append(
            Violation(
                code="unknown_reference",
                message=f"victim {case.victim!r} is not a person",
            )
        )
    if case.crime_location not in location_ids:
        out.append(
            Violation(
                code="unknown_reference",
                message=f"crime_location {case.crime_location!r} is not a location",
            )
        )
    for event in case.timeline:
        if event.location not in location_ids:
            out.append(
                Violation(
                    code="unknown_reference",
                    message=(
                        f"timeline event {event.id!r} is at unknown location {event.location!r}"
                    ),
                )
            )
        for person_id in [*event.actors, *event.witnesses]:
            if person_id not in person_ids:
                out.append(
                    Violation(
                        code="unknown_reference",
                        message=(
                            f"timeline event {event.id!r} references unknown person {person_id!r}"
                        ),
                    )
                )
    for clue in case.clues:
        if clue.location not in location_ids:
            out.append(
                Violation(
                    code="unknown_reference",
                    message=f"clue {clue.id!r} is at unknown location {clue.location!r}",
                )
            )
        if clue.points_to is not None and clue.points_to not in person_ids:
            out.append(
                Violation(
                    code="unknown_reference",
                    message=f"clue {clue.id!r} points to unknown person {clue.points_to!r}",
                )
            )
    return out


def validate(case: Case) -> list[Violation]:
    """Run every check and return all violations (empty when the case is sound)."""
    violations: list[Violation] = []
    violations += _check_references(case)
    violations += _check_murderer(case)
    violations += _check_murderer_at_scene(case)
    violations += _check_double_booked(case)
    violations += _check_secrets(case)
    violations += _check_pointing_clues(case)
    violations += _check_suspicion_spread(case)
    return violations
