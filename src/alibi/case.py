"""The case file: Pydantic models for one mystery.

These are the *contract* the generator must satisfy and the validator checks.
Times are ``"HH:MM"`` strings (readable in prompts, sortable in code).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def parse_time(value: str) -> int:
    """``"22:00"`` -> minutes since midnight. Raises on bad input."""
    match = _TIME_RE.match(value)
    if not match:
        raise ValueError(f"time must be 'HH:MM' (24h), got {value!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def format_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class Secret(BaseModel):
    """Something a person hides. Some are red herrings, unrelated to the crime."""

    id: str
    text: str
    is_red_herring: bool = False


class Person(BaseModel):
    id: str
    name: str
    description: str = ""
    is_victim: bool = False
    secrets: list[Secret] = Field(default_factory=list)


class Location(BaseModel):
    id: str
    name: str
    description: str = ""


class TimelineEvent(BaseModel):
    id: str
    time: str
    location: str
    actors: list[str] = Field(default_factory=list)
    witnesses: list[str] = Field(default_factory=list)
    description: str = ""

    @field_validator("time")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        parse_time(value)
        return value

    @property
    def minutes(self) -> int:
        return parse_time(self.time)


class Clue(BaseModel):
    id: str
    location: str
    description: str
    points_to: str | None = None
    is_red_herring: bool = False


class Case(BaseModel):
    """One complete, self-contained mystery."""

    id: str
    title: str
    setting: str = ""
    time_of_death: str
    crime_location: str
    murderer: str
    victim: str
    persons: list[Person] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    clues: list[Clue] = Field(default_factory=list)

    @field_validator("time_of_death")
    @classmethod
    def _validate_time_of_death(cls, value: str) -> str:
        parse_time(value)
        return value

    # -- lookups ------------------------------------------------------------
    def person(self, person_id: str) -> Person | None:
        return next((p for p in self.persons if p.id == person_id), None)

    def location(self, location_id: str) -> Location | None:
        return next((loc for loc in self.locations if loc.id == location_id), None)

    def clue(self, clue_id: str) -> Clue | None:
        return next((c for c in self.clues if c.id == clue_id), None)

    # -- derived views ------------------------------------------------------
    def suspects(self) -> list[Person]:
        """Everyone except the victim."""
        return [p for p in self.persons if not p.is_victim]

    def victim_person(self) -> Person | None:
        return next((p for p in self.persons if p.is_victim), None)
