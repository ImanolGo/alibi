"""The game: state, actions, scoring and persistence.

The player — a human in the terminal or an MCP client — issues a fixed set of
actions (search, inspect, question, lie detector, accuse). This module owns
what each action does and what it costs. Costly actions run through a small
LangGraph pipeline (validate → apply → update); informational ones are free.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .case import Case
from .db import Game as GameRow
from .db import get_session_factory
from .llm import LLM
from .suspect import SuspectTeam

log = logging.getLogger(__name__)

MAX_ACTIONS = 20
MAX_LIE_DETECTOR = 2


class ActionResult(BaseModel):
    """What an action produced. ``ok=False`` is a friendly, expected error."""

    ok: bool = True
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class GameState(BaseModel):
    """Everything mutable about a game. JSON-serialisable (stored in Postgres)."""

    game_id: str
    case_id: str
    actions_left: int = MAX_ACTIONS
    discovered_clues: list[str] = Field(default_factory=list)
    examined_clues: list[str] = Field(default_factory=list)
    lie_detector_used: int = 0
    conversations: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    accused: str | None = None
    outcome: Literal["playing", "won", "lost"] = "playing"
    log: list[str] = Field(default_factory=list)


class Responder(Protocol):
    """The bit of the game that talks to suspects (faked in tests)."""

    def set_clues(self, suspect_id: str, clue_ids: list[str]) -> None: ...

    def answer(self, suspect_id: str, question: str) -> str: ...


class ActionState(TypedDict, total=False):
    action: dict[str, Any]
    result: ActionResult


class GameStore(Protocol):
    def save(self, game: Game) -> None: ...

    def load(self, game_id: str) -> Game | None: ...


class InMemoryGameStore:
    """Store for tests and dry runs; keeps rows in a dict."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def save(self, game: Game) -> None:
        self.rows[game.game_id] = game.to_row()

    def load(self, game_id: str) -> Game | None:
        row = self.rows.get(game_id)
        return None if row is None else Game.from_row(row, store=self)


class SqlGameStore:
    """Postgres-backed store: a game survives an app restart."""

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def save(self, game: Game) -> None:
        row = game.to_row()
        with self._session_factory() as session:
            existing = session.get(GameRow, row["game_id"])
            if existing is None:
                session.add(GameRow(**row))
            else:
                existing.case_id = row["case_id"]
                existing.state = row["state"]
                existing.case_json = row["case_json"]
            session.commit()

    def load(self, game_id: str) -> Game | None:
        with self._session_factory() as session:
            row = session.get(GameRow, game_id)
            if row is None:
                return None
            data = {
                "game_id": row.game_id,
                "case_id": row.case_id,
                "state": row.state,
                "case_json": row.case_json,
            }
        return Game.from_row(data, store=self)


class Game:
    """One playable game over one case."""

    def __init__(
        self,
        case: Case,
        *,
        game_id: str | None = None,
        store: GameStore | None = None,
        responder: Responder | None = None,
        llm: LLM | None = None,
        reason: bool | None = None,
    ) -> None:
        self.case = case
        self.game_id = game_id or case.id
        self.store = store
        self.responder = responder or SuspectTeam(
            case, game_id=self.game_id, llm=llm, reason=reason
        )
        self.state = GameState(game_id=self.game_id, case_id=case.id)
        self._pipeline = self._build_pipeline()
        if self.store is not None:
            self.store.save(self)

    # -- persistence --------------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "case_id": self.case.id,
            "state": self.state.model_dump(),
            "case_json": self.case.model_dump(),
        }

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        *,
        store: GameStore | None = None,
        responder: Responder | None = None,
        llm: LLM | None = None,
        reason: bool | None = None,
    ) -> Game:
        case = Case.model_validate(row["case_json"])
        game = cls(
            case,
            game_id=row["game_id"],
            store=store,
            responder=responder,
            llm=llm,
            reason=reason,
        )
        game.state = GameState.model_validate(row["state"])
        return game

    def save(self) -> None:
        if self.store is not None:
            self.store.save(self)

    # -- free, informational actions ---------------------------------------
    def list_rooms(self) -> ActionResult:
        rooms = [
            {"id": loc.id, "name": loc.name, "description": loc.description}
            for loc in self.case.locations
        ]
        message = "\n".join(
            f"- {room['name']} ({room['id']}): {room['description']}" for room in rooms
        )
        return ActionResult(message=message, data={"rooms": rooms})

    def case_summary(self) -> ActionResult:
        suspects = [p.id for p in self.case.suspects()]
        return ActionResult(
            message=f"{self.case.title}. The victim is {self.case.victim}.",
            data={"title": self.case.title, "suspects": suspects, "victim": self.case.victim},
        )

    def status(self) -> ActionResult:
        return ActionResult(
            message=(
                f"{self.state.actions_left} actions left. "
                f"You have found {len(self.state.discovered_clues)} clue(s). "
                f"Outcome: {self.state.outcome}."
            ),
            data=self.state.model_dump(),
        )

    # -- costly actions (through the pipeline) ------------------------------
    def search_room(self, room_id: str) -> ActionResult:
        return self._run("search", room_id=room_id)

    def inspect(self, clue_id: str) -> ActionResult:
        return self._run("inspect", clue_id=clue_id)

    def question(self, suspect_id: str, text: str) -> ActionResult:
        return self._run("question", suspect_id=suspect_id, text=text)

    def lie_detector(self) -> ActionResult:
        return self._run("detector")

    def accuse(self, suspect_id: str, motive: str | None = None) -> ActionResult:
        return self._run("accuse", suspect_id=suspect_id, motive=motive)

    # -- pipeline -----------------------------------------------------------
    def _build_pipeline(self) -> Any:
        graph = StateGraph(ActionState)
        graph.add_node("validate", self._validate_node)
        graph.add_node("apply", self._apply_node)
        graph.add_node("update", self._update_node)
        graph.add_edge(START, "validate")
        graph.add_edge("validate", "apply")
        graph.add_edge("apply", "update")
        graph.add_edge("update", END)
        return graph.compile()

    def _run(self, kind: str, **params: Any) -> ActionResult:
        final = self._pipeline.invoke({"action": {"kind": kind, **params}})
        result = final.get("result")
        if result is None:  # pragma: no cover - defensive
            return ActionResult(ok=False, message="That action could not be processed.")
        return result

    def _cost(self, kind: str) -> int:
        # The lie detector is a stub until M5; do not charge the player for it.
        return 0 if kind == "detector" else 1

    def _validate_node(self, state: ActionState) -> dict[str, Any]:
        action = state["action"]
        if self.state.outcome != "playing":
            return {
                "result": ActionResult(
                    ok=False, message=f"The game is over ({self.state.outcome})."
                )
            }
        if self.state.actions_left <= 0:
            return {"result": ActionResult(ok=False, message="You have no actions left.")}
        error = self._validate_action(action)
        if error is not None:
            return {"result": ActionResult(ok=False, message=error)}
        return {}

    def _apply_node(self, state: ActionState) -> dict[str, Any]:
        result = state.get("result")
        if result is not None and not result.ok:
            return {}
        return {"result": self._apply_action(state["action"])}

    def _update_node(self, state: ActionState) -> dict[str, Any]:
        result = state.get("result")
        if result is None:
            result = ActionResult(ok=False, message="That action could not be processed.")
        if result.ok:
            kind = state["action"]["kind"]
            self.state.actions_left -= self._cost(kind)
            self.state.log.append(kind)
            if self.state.actions_left <= 0 and self.state.outcome == "playing":
                self.state.outcome = "lost"
                result.data.setdefault("outcome", "lost")
            self.save()
        return {"result": result}

    # -- validation ---------------------------------------------------------
    def _validate_action(self, action: dict[str, Any]) -> str | None:
        kind = action["kind"]
        if kind == "search":
            if self.case.location(action["room_id"]) is None:
                rooms = ", ".join(loc.id for loc in self.case.locations)
                return f"No such room {action['room_id']!r}. Rooms: {rooms}."
        elif kind == "inspect":
            if self.case.clue(action["clue_id"]) is None:
                return f"No such clue {action['clue_id']!r}."
            if action["clue_id"] not in self.state.discovered_clues:
                return "You have not found that yet — search a room first."
        elif kind == "question":
            if self.case.person(action["suspect_id"]) is None:
                return f"No such suspect {action['suspect_id']!r}."
            if not action.get("text", "").strip():
                return "You must actually ask something."
        elif kind == "accuse":
            if self.case.person(action["suspect_id"]) is None:
                return f"No such suspect {action['suspect_id']!r}."
        elif kind == "detector" and self.state.lie_detector_used >= MAX_LIE_DETECTOR:
            return "The lie detector has already been used twice."
        return None

    # -- applying -----------------------------------------------------------
    def _apply_action(self, action: dict[str, Any]) -> ActionResult:
        kind = action["kind"]
        if kind == "search":
            return self._search(action["room_id"])
        if kind == "inspect":
            return self._inspect(action["clue_id"])
        if kind == "question":
            return self._question(action["suspect_id"], action["text"])
        if kind == "detector":
            return self._detector()
        if kind == "accuse":
            return self._accuse(action["suspect_id"], action.get("motive"))
        return ActionResult(ok=False, message=f"Unknown action {kind!r}.")  # pragma: no cover

    def _search(self, room_id: str) -> ActionResult:
        room = self.case.location(room_id)
        assert room is not None  # validated
        found = [clue for clue in self.case.clues if clue.location == room_id]
        newly = [clue.id for clue in found if clue.id not in self.state.discovered_clues]
        self.state.discovered_clues.extend(newly)

        if not found:
            return ActionResult(message=f"You search {room.name} but find nothing of note.")
        lines = [f"You search {room.name} and notice:"]
        lines += [f"- {clue.description} (clue id: {clue.id})" for clue in found]
        return ActionResult(
            message="\n".join(lines),
            data={"clues": [clue.id for clue in found], "new": newly},
        )

    def _inspect(self, clue_id: str) -> ActionResult:
        clue = self.case.clue(clue_id)
        assert clue is not None  # validated
        if clue_id not in self.state.examined_clues:
            self.state.examined_clues.append(clue_id)
        hint = ""
        if clue.points_to:
            person = self.case.person(clue.points_to)
            if person is not None:
                hint = f" It casts suspicion on {person.name}."
        return ActionResult(
            message=f"Looking closely at {clue_id}: {clue.description}{hint}",
            data={"points_to": clue.points_to, "red_herring": clue.is_red_herring},
        )

    def _question(self, suspect_id: str, text: str) -> ActionResult:
        self.responder.set_clues(suspect_id, self.state.discovered_clues)
        answer = self.responder.answer(suspect_id, text)
        thread = self.state.conversations.setdefault(suspect_id, [])
        thread.append({"role": "detective", "content": text})
        thread.append({"role": "suspect", "content": answer})
        return ActionResult(message=answer, data={"suspect": suspect_id})

    def _detector(self) -> ActionResult:
        self.state.lie_detector_used += 1
        return ActionResult(
            message="The lie-detector needle twitches and settles. It is not ready yet (M5).",
            data={"available": False},
        )

    def _accuse(self, suspect_id: str, motive: str | None) -> ActionResult:
        self.state.accused = suspect_id
        correct = suspect_id == self.case.murderer
        self.state.outcome = "won" if correct else "lost"
        murderer = self.case.person(self.case.murderer)
        murderer_name = murderer.name if murderer else self.case.murderer
        accused = self.case.person(suspect_id)
        accused_name = accused.name if accused else suspect_id
        verdict = "Correct!" if correct else "Wrong."
        reveal = f"The murderer was {murderer_name}."
        if motive:
            reveal += f" Motive given: {motive}."
        return ActionResult(
            message=f"You accuse {accused_name}. {verdict} {reveal}",
            data={"correct": correct, "murderer": self.case.murderer, "motive": motive},
        )


def load_or_create_game(
    case: Case,
    *,
    game_id: str,
    store: GameStore,
    responder: Responder | None = None,
    llm: LLM | None = None,
    reason: bool | None = None,
) -> Game:
    """Resume a saved game, or start a new one and persist it."""
    existing = store.load(game_id)
    if existing is not None:
        return existing
    return Game(
        case,
        game_id=game_id,
        store=store,
        responder=responder,
        llm=llm,
        reason=reason,
    )
