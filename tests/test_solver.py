from __future__ import annotations

from alibi.case import Case
from alibi.fake_llm import FakeLLM
from alibi.game import Game, InMemoryGameStore
from alibi.solver import SolverAction, SolverAgent


class FakeResponder:
    def set_clues(self, suspect_id: str, clue_ids: list[str]) -> None: ...

    def answer(self, suspect_id: str, question: str) -> str:
        return f"{suspect_id} answers evasively."


def make_solver(case: Case, actions: list[SolverAction]) -> tuple[SolverAgent, Game]:
    game = Game(case, store=InMemoryGameStore(), responder=FakeResponder())
    solver = SolverAgent(game, llm=FakeLLM(structured=list(actions)))
    return solver, game


# --- the solver cannot see the ground truth --------------------------------
def test_solver_context_hides_secrets_and_the_label(valid_case: Case) -> None:
    solver, _ = make_solver(valid_case, [])
    text = "\n".join(message["content"] for message in solver.build_messages())

    for person in valid_case.persons:
        for secret in person.secrets:
            assert secret.text not in text
    # The solver is never told that anyone is "the murderer".
    assert "murderer" not in text.lower()
    assert "culprit" not in text.lower()


# --- a scripted play -------------------------------------------------------
def test_solver_plays_a_full_game(valid_case: Case) -> None:
    actions = [
        SolverAction(kind="search_room", room_id="library"),
        SolverAction(kind="inspect", clue_id="clue_ash"),
        SolverAction(kind="question", suspect_id="butler_hobbs", text="Where were you?"),
        SolverAction(kind="accuse", suspect_id="butler_hobbs", motive="the ledger"),
    ]
    solver, game = make_solver(valid_case, actions)

    transcript = solver.play()

    assert game.state.outcome == "won"
    assert len(transcript) == 4
    assert transcript[0][0].startswith("search_room")
    assert "Correct!" in transcript[-1][1]


def test_invalid_action_is_reported_and_the_game_continues(valid_case: Case) -> None:
    actions = [
        SolverAction(kind="search_room"),  # missing room_id
        SolverAction(kind="accuse", suspect_id="butler_hobbs"),
    ]
    solver, game = make_solver(valid_case, actions)

    transcript = solver.play()

    assert "Invalid action" in transcript[0][1]
    assert game.state.outcome == "won"


def test_solver_stops_at_max_steps(valid_case: Case) -> None:
    actions = [SolverAction(kind="status") for _ in range(3)]
    game = Game(valid_case, store=InMemoryGameStore(), responder=FakeResponder())
    solver = SolverAgent(game, llm=FakeLLM(structured=actions), max_steps=3)

    transcript = solver.play()

    assert len(transcript) == 3
    assert game.state.outcome == "playing"
