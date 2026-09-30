"""The suspect agent: a LangGraph of retrieve → think → speak → remember.

Information hiding lives here. The agent only ever sees its own persona, its
own secrets, what ``knowledge.py`` says it knows, and its own memories — never
the full case and never another suspect's secret. For an innocent, the prompt
never even mentions a "murderer".
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .case import Case, Clue, TimelineEvent
from .config import load_setting
from .generator import render
from .knowledge import knowledge_for
from .llm import LLM, Message, default_client
from .memory import MemoryStore, recall, remember
from .tracing import span

log = logging.getLogger(__name__)

# How many recent turns of the conversation to include verbatim.
MAX_HISTORY_TURNS = 6


class Thought(BaseModel):
    """The suspect's private reasoning. Shown only with ``--debug``."""

    worry_level: int = 0
    strategy: str = ""
    facts_to_use: list[str] = []


class SuspectState(TypedDict, total=False):
    """LangGraph state for a single turn (kept minimal and serialisable)."""

    question: str
    retrieved: list[str]
    thought: Thought
    answer: str


class SuspectAgent:
    """One suspect you can interrogate, turn by turn."""

    def __init__(
        self,
        case: Case,
        person_id: str,
        *,
        store: MemoryStore,
        llm: LLM | None = None,
        game_id: str = "local",
        setting: dict[str, Any] | None = None,
    ) -> None:
        person = case.person(person_id)
        if person is None:
            raise KeyError(f"unknown person {person_id!r}")
        self.case = case
        self.person = person
        self.person_id = person_id
        self.game_id = game_id
        self.store = store
        self.llm = llm or default_client()
        self.setting = setting if setting is not None else load_setting()
        self.is_murderer = case.murderer == person_id
        self.clues_shown: list[str] = []
        self.history: list[Message] = []
        self.last_thought: Thought | None = None
        self.graph = self._build_graph()

    # -- public API ---------------------------------------------------------
    def answer(self, question: str) -> str:
        """Run one turn of the graph and return the suspect's reply."""
        with span(
            "suspect.turn",
            **{"suspect.id": self.person_id, "game.id": self.game_id},
        ):
            state = self.graph.invoke({"question": question, "retrieved": []})
        answer = state.get("answer", "")
        self.last_thought = state.get("thought")
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def show_clue(self, clue_id: str) -> str:
        """Confront the suspect with a clue and let them react."""
        clue = self.case.clue(clue_id)
        if clue is None:
            raise KeyError(f"unknown clue {clue_id!r}")
        if clue_id not in self.clues_shown:
            self.clues_shown.append(clue_id)
        return self.answer(
            "The detective sets an item before you and says nothing. "
            f"The item is: {clue.description} How do you react?"
        )

    def pointing_clues_shown(self) -> int:
        """Genuine clues against *this* suspect that the detective has shown."""
        count = 0
        for clue_id in self.clues_shown:
            clue = self.case.clue(clue_id)
            if clue and not clue.is_red_herring and clue.points_to == self.person_id:
                count += 1
        return count

    # -- prompt context (information hiding) --------------------------------
    def system_prompt(self) -> str:
        """The persona block. This is what a test asserts stays secret-free."""
        knowledge = knowledge_for(self.case, self.person_id)
        return render(
            "suspect_system.jinja",
            person=self.person,
            setting=self.setting,
            victim_name=self._victim_name(),
            known_events=[self._event_view(e) for e in knowledge.events],
            own_secrets=knowledge.secrets,
            is_murderer=self.is_murderer,
            pointing_clues_shown=self.pointing_clues_shown(),
        )

    def _victim_name(self) -> str:
        victim = self.case.victim_person()
        return victim.name if victim else "the victim"

    def _names(self, person_ids: list[str]) -> list[str]:
        out = []
        for person_id in person_ids:
            person = self.case.person(person_id)
            out.append(person.name if person else person_id)
        return out

    def _event_view(self, event: TimelineEvent) -> dict[str, Any]:
        location = self.case.location(event.location)
        return {
            "time": event.time,
            "location": location.name if location else event.location,
            "actors": self._names(event.actors),
            "witnesses": self._names(event.witnesses),
            "description": event.description,
        }

    def _clue_views(self) -> list[dict[str, Any]]:
        views = []
        for clue_id in self.clues_shown:
            clue = self.case.clue(clue_id)
            if clue is not None:
                views.append(self._clue_view(clue))
        return views

    @staticmethod
    def _clue_view(clue: Clue) -> dict[str, Any]:
        return {
            "id": clue.id,
            "description": clue.description,
            "is_red_herring": clue.is_red_herring,
        }

    # -- graph nodes --------------------------------------------------------
    def _retrieve(self, state: SuspectState) -> dict[str, Any]:
        with span("suspect.retrieve", **{"suspect.id": self.person_id}):
            hits = recall(
                self.store,
                game_id=self.game_id,
                suspect_id=self.person_id,
                query=state["question"],
                llm=self.llm,
            )
        return {"retrieved": [hit.content for hit in hits]}

    def _think(self, state: SuspectState) -> dict[str, Any]:
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt()},
            {
                "role": "user",
                "content": render(
                    "suspect_think.jinja",
                    question=state["question"],
                    retrieved=state.get("retrieved", []),
                ),
            },
        ]
        with span("suspect.think", **{"suspect.id": self.person_id}):
            thought = self.llm.structured(messages, Thought, role="suspect")
        return {"thought": thought}

    def _speak(self, state: SuspectState) -> dict[str, Any]:
        thought = state.get("thought") or Thought()
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt()},
            {
                "role": "user",
                "content": render(
                    "suspect_speak.jinja",
                    person=self.person,
                    question=state["question"],
                    retrieved=state.get("retrieved", []),
                    clues=self._clue_views(),
                    thought=thought,
                    history=self.history[-MAX_HISTORY_TURNS * 2 :],
                ),
            },
        ]
        with span("suspect.speak", **{"suspect.id": self.person_id}):
            answer = self.llm.complete(messages, role="suspect")
        return {"answer": answer}

    def _remember(self, state: SuspectState) -> dict[str, Any]:
        with span("suspect.remember", **{"suspect.id": self.person_id}):
            remember(
                self.store,
                game_id=self.game_id,
                suspect_id=self.person_id,
                content=(
                    f"Detective asked: {state['question']}\nI answered: {state.get('answer', '')}"
                ),
                llm=self.llm,
            )
        return {}

    def _build_graph(self) -> Any:
        graph = StateGraph(SuspectState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("think", self._think)
        graph.add_node("speak", self._speak)
        graph.add_node("remember", self._remember)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "think")
        graph.add_edge("think", "speak")
        graph.add_edge("speak", "remember")
        graph.add_edge("remember", END)
        return graph.compile()
