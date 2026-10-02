from __future__ import annotations

from alibi.case import Case
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


# --- lie detector (stub) ---------------------------------------------------
def test_lie_detector_is_free_and_capped_at_two(valid_case: Case) -> None:
    game, _ = make_game(valid_case)
    assert game.lie_detector().ok
    assert game.lie_detector().ok
    third = game.lie_detector()
    assert not third.ok and "twice" in third.message
    assert game.state.actions_left == MAX_ACTIONS


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
