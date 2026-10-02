"""The automated detective: a LangGraph that plays through the game tools.

It sees only what a player sees — the case summary, the rooms, the status and
the results of its own actions. It never sees the case's ground truth, so it
cannot cheat. This agent exists only for evals (M4).
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .game import ActionResult, Game
from .generator import render
from .llm import LLM, Message, default_client

log = logging.getLogger(__name__)

MAX_STEPS = 25


class SolverAction(BaseModel):
    """The detective's next move."""

    kind: Literal["search_room", "inspect", "question", "status", "accuse"]
    room_id: str | None = None
    clue_id: str | None = None
    suspect_id: str | None = None
    text: str | None = None
    evidence: list[str] | None = None
    motive: str | None = None
    reasoning: str = ""


class SolverState(TypedDict, total=False):
    action: SolverAction
    result: ActionResult
    done: bool
    steps: int


class SolverAgent:
    """A LangGraph (plan → act → repeat) that plays one :class:`Game`."""

    def __init__(self, game: Game, *, llm: LLM | None = None, max_steps: int = MAX_STEPS) -> None:
        self.game = game
        self.llm = llm or default_client()
        self.max_steps = max_steps
        self.transcript: list[tuple[str, str]] = []
        self.actions: list[SolverAction] = []
        self.graph = self._build_graph()

    # -- context (what the solver is allowed to see) ------------------------
    def context(self) -> dict[str, Any]:
        return {
            "case_summary": self.game.case_summary().message,
            "rooms": self.game.list_rooms().message,
            "status": self.game.status().message,
            "transcript": [
                {"action": action, "result": result} for action, result in self.transcript
            ],
        }

    def build_messages(self) -> list[Message]:
        return [
            {"role": "system", "content": render("solver_system.jinja")},
            {"role": "user", "content": render("solver_plan.jinja", **self.context())},
        ]

    # -- graph --------------------------------------------------------------
    def _build_graph(self) -> Any:
        graph = StateGraph(SolverState)
        graph.add_node("plan", self._plan)
        graph.add_node("act", self._act)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "act")
        graph.add_conditional_edges("act", self._route, {"plan": "plan", "end": END})
        return graph.compile()

    def _plan(self, state: SolverState) -> dict[str, Any]:
        action = self.llm.structured(self.build_messages(), SolverAction, role="solver")
        return {"action": action}

    def _act(self, state: SolverState) -> dict[str, Any]:
        action = state["action"]
        result = self._dispatch(action)
        self.actions.append(action)
        self.transcript.append((self._describe(action), result.message))
        log.info(
            "solver step %d: %s -> %s",
            len(self.actions),
            self._describe(action),
            result.message.splitlines()[0] if result.message else "",
        )
        done = self.game.state.outcome != "playing" or len(self.actions) >= self.max_steps
        return {"result": result, "done": done, "steps": len(self.actions)}

    def _route(self, state: SolverState) -> str:
        return "end" if state.get("done") else "plan"

    def _dispatch(self, action: SolverAction) -> ActionResult:
        if action.kind == "search_room" and action.room_id:
            return self.game.search_room(action.room_id)
        if action.kind == "inspect" and action.clue_id:
            return self.game.inspect(action.clue_id)
        if action.kind == "question" and action.suspect_id and action.text:
            return self.game.question(action.suspect_id, action.text, action.evidence)
        if action.kind == "accuse" and action.suspect_id:
            return self.game.accuse(action.suspect_id, action.motive)
        if action.kind == "status":
            return self.game.status()
        return ActionResult(
            ok=False,
            message=(
                f"Invalid action {action.kind!r}: missing arguments. "
                "Give the required ids (room_id, clue_id, suspect_id, text)."
            ),
        )

    @staticmethod
    def _describe(action: SolverAction) -> str:
        parts: list[str] = [action.kind]
        for field in ("room_id", "clue_id", "suspect_id", "text", "evidence", "motive"):
            value = getattr(action, field)
            if value:
                parts.append(f"{field}={value!r}")
        return " ".join(parts)

    def play(self) -> list[tuple[str, str]]:
        """Run the whole game and return the (action, result) transcript."""
        self.graph.invoke({})
        return self.transcript
