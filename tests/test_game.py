from __future__ import annotations

from alibi.case import Case
from alibi.detector import Prediction
from alibi.game import MAX_ACTIONS, Game, InMemoryGameStore


class FakeResponder:
    def __init__(self, reply: str = "No comment.") -> None:
        self.reply = reply
        self.clues: dict[str, list[str]] = {}
        self.calls: list[tuple[str, str]] = []

    def set_clues(self, suspect_id: str, clue_ids: list[str]) -> None:
        self.clues[suspect_id] = list(clue_ids)

    def answer(self, suspect_id: str, question: str) -> str:
        self.calls.append((suspect_id, question))
        return f"{suspect_id}: {self.reply}"


def make_game(case: Case, *, store: InMemoryGameStore | None = None) -> tuple[Game, FakeResponder]:
    responder = FakeResponder()
    return Game(case, store=store or InMemoryGameStore(), responder=responder), responder


# --- action counting -------------------------------------------------------
def test_search_costs_one_action(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    result = game.search_room("library")
    assert result.ok
    assert game.state.actions_left == MAX_ACTIONS - 1


def test_invalid_action_is_free(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    result = game.search_room("atlantis")
    assert not result.ok
    assert "No such room" in result.message
    assert game.state.actions_left == MAX_ACTIONS


def test_free_actions_do_not_count(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    game.list_rooms()
    game.case_summary()
    game.status()
    assert game.state.actions_left == MAX_ACTIONS


def test_running_out_of_actions_loses(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    for _ in range(MAX_ACTIONS):
        assert game.search_room("conservatory").ok
    assert game.state.actions_left == 0
    assert game.state.outcome == "lost"
    later = game.search_room("conservatory")
    assert not later.ok and "over" in later.message


# --- clue discovery rules --------------------------------------------------
def test_search_discovers_the_rooms_clues(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    result = game.search_room("library")
    assert set(result.data["clues"]) == {"clue_ash", "clue_cufflink"}
    assert set(game.state.discovered_clues) == {"clue_ash", "clue_cufflink"}


def test_inspect_requires_discovery_first(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    before = game.inspect("clue_ash")
    assert not before.ok and "search a room first" in before.message

    game.search_room("library")
    after = game.inspect("clue_ash")
    assert after.ok
    assert "butler's ash" in after.message  # the evidence is described
    assert "Hobbs" not in after.message  # but the suspect is never named (fair play)


def test_searching_an_empty_room_is_fine(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    result = game.search_room("kitchen")  # no clues placed here
    assert result.ok
    assert "nothing of note" in result.message


# --- questioning -----------------------------------------------------------
def test_question_records_thread_but_alone_shows_no_clues(valid_case: Case) -> None:
    game, responder = make_game(valid_case)
    game.search_room("library")
    result = game.question("lady_blackwood", "Where were you at ten?")

    assert result.ok
    assert responder.calls == [("lady_blackwood", "Where were you at ten?")]
    # Finding a clue is not the same as showing it.
    assert responder.clues["lady_blackwood"] == []
    assert game.state.presented.get("lady_blackwood", []) == []
    thread = game.state.conversations["lady_blackwood"]
    assert thread[0] == {"role": "detective", "content": "Where were you at ten?"}
    assert thread[1]["role"] == "suspect"
    assert game.state.actions_left == MAX_ACTIONS - 2


def test_presenting_evidence_shows_only_what_is_presented(valid_case: Case) -> None:
    game, responder = make_game(valid_case)
    game.search_room("library")  # clue_ash, clue_cufflink
    result = game.question("lady_blackwood", "Explain this.", evidence=["clue_ash"])

    assert result.ok
    assert result.data["evidence_presented"] == ["clue_ash"]
    assert responder.clues["lady_blackwood"] == ["clue_ash"]


def test_evidence_must_be_discovered_first(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    result = game.question("lady_blackwood", "Explain.", evidence=["clue_ash"])
    assert not result.ok and "not found" in result.message
    assert game.state.actions_left == MAX_ACTIONS


def test_case_summary_lists_suspects(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    result = game.case_summary()
    assert result.ok
    assert "butler_hobbs" in result.message
    assert "Hobbs" in result.message  # id and name, so a solver can accuse by id


def test_unknown_suspect_is_a_friendly_error(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    result = game.question("the_gardener", "Hello?")
    assert not result.ok and "No such suspect" in result.message
    assert game.state.actions_left == MAX_ACTIONS


# --- lie detector ----------------------------------------------------------
class FakeDetector:
    """Says every statement containing 'kitchen' is a lie."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def predict(self, statement: str) -> Prediction:
        self.statements.append(statement)
        lie = "kitchen" in statement
        return Prediction(label="FALSE" if lie else "TRUE", probability=0.9 if lie else 0.1)


def make_detector_game(case: Case) -> tuple[Game, FakeDetector]:
    detector = FakeDetector()
    game = Game(case, store=InMemoryGameStore(), responder=FakeResponder(), detector=detector)
    return game, detector


def test_lie_detector_is_free_and_capped_at_two(valid_case: Case) -> None:
    game, _ = make_detector_game(valid_case)
    game.question("butler_hobbs", "Where were you?")
    assert game.lie_detector().ok
    assert game.lie_detector().ok
    third = game.lie_detector()
    assert not third.ok and "twice" in third.message
    assert game.state.actions_left == MAX_ACTIONS - 1  # only the question cost anything


def test_lie_detector_with_nothing_to_test_keeps_its_charge(valid_case: Case) -> None:
    game, detector = make_detector_game(valid_case)
    result = game.lie_detector()
    assert not result.ok and "question someone first" in result.message
    assert game.state.lie_detector_used == 0
    assert detector.statements == []


def test_missing_detector_model_keeps_its_charge(valid_case: Case) -> None:
    game, _ = make_game(valid_case)  # no model on disk in tests
    game.question("butler_hobbs", "Where were you?")
    result = game.lie_detector()
    assert result.data["available"] is False
    assert game.state.lie_detector_used == 0


def test_lie_detector_tests_the_most_recent_answer(valid_case: Case) -> None:
    game, detector = make_detector_game(valid_case)
    game.question("lady_blackwood", "Where were you?")
    game.question("butler_hobbs", "Where were you?")
    game.question("lady_blackwood", "And after that, in the kitchen?")
    game.lie_detector()
    # Lady Blackwood answered last, even though Hobbs's thread was opened later.
    assert detector.statements[-1].startswith("lady_blackwood")
    assert game.lie_detector("butler_hobbs").ok
    assert detector.statements[-1].startswith("butler_hobbs")


# --- who can be questioned -------------------------------------------------
def test_the_victim_cannot_be_questioned_or_accused(valid_case: Case) -> None:
    game, responder = make_game(valid_case)
    asked = game.question("lord_blackwood", "Who did it?")
    assert not asked.ok and "No such suspect" in asked.message
    accused = game.accuse("lord_blackwood")
    assert not accused.ok
    assert responder.calls == [] and game.state.outcome == "playing"


# --- accusation / scoring --------------------------------------------------
def test_accusing_the_murderer_wins(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    result = game.accuse("butler_hobbs")
    assert result.ok and result.data["correct"] is True
    assert game.state.outcome == "won"


def test_accusing_the_wrong_person_loses(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    result = game.accuse("dr_penrose", motive="gambling debts")
    assert result.data["correct"] is False
    assert game.state.outcome == "lost"
    assert "butler_hobbs" in result.message or "Hobbs" in result.message


def test_game_over_blocks_further_actions(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    game.accuse("butler_hobbs")
    blocked = game.search_room("library")
    assert not blocked.ok and "over" in blocked.message


# --- persistence -----------------------------------------------------------
def test_game_survives_a_restart(valid_case: Case) -> None:
    store = InMemoryGameStore()
    game, _ = make_game(valid_case, store=store)
    game.search_room("library")
    game.question("butler_hobbs", "Where were you?")
    game_id = game.game_id

    reloaded = store.load(game_id)
    assert reloaded is not None
    assert reloaded.case.id == valid_case.id
    assert reloaded.state.discovered_clues == ["clue_ash", "clue_cufflink"]
    assert reloaded.state.actions_left == MAX_ACTIONS - 2
    assert "butler_hobbs" in reloaded.state.conversations


def test_loading_a_game_does_not_overwrite_the_saved_state(valid_case: Case) -> None:
    store = InMemoryGameStore()
    game, _ = make_game(valid_case, store=store)
    game.search_room("library")

    store.load(game.game_id)  # e.g. a page view after a restart
    again = store.load(game.game_id)

    assert again is not None
    assert again.state.actions_left == MAX_ACTIONS - 1
    assert store.rows[game.game_id]["state"]["actions_left"] == MAX_ACTIONS - 1


class HistoryResponder(FakeResponder):
    def __init__(self) -> None:
        super().__init__()
        self.restored: dict[str, list[dict[str, str]]] = {}

    def restore(self, conversations: dict[str, list[dict[str, str]]]) -> None:
        self.restored = conversations


def test_reloaded_suspects_remember_the_conversation(valid_case: Case) -> None:
    store = InMemoryGameStore()
    game, _ = make_game(valid_case, store=store)
    game.question("butler_hobbs", "Where were you?")

    responder = HistoryResponder()
    row = store.rows[game.game_id]
    Game.from_row(row, store=store, responder=responder)

    assert responder.restored["butler_hobbs"][0] == {
        "role": "detective",
        "content": "Where were you?",
    }
