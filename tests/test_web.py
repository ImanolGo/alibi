from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alibi.case import Case
from alibi.game import MAX_ACTIONS, Game, InMemoryGameStore
from alibi.web import (
    case_review,
    clue_cards,
    create_app,
    detective_rank,
    friendly,
    story_clock,
)


class FakeResponder:
    def __init__(self) -> None:
        self.clues: dict[str, list[str]] = {}

    def set_clues(self, suspect_id: str, clue_ids: list[str]) -> None:
        self.clues[suspect_id] = list(clue_ids)

    def answer(self, suspect_id: str, question: str) -> str:
        return f"{suspect_id}: I was in the kitchen."


def make_client(tmp_path: Path, case: Case, **kwargs) -> TestClient:
    (tmp_path / f"{case.id}.json").write_text(case.model_dump_json(), encoding="utf-8")
    kwargs.setdefault("game_store", InMemoryGameStore())
    kwargs.setdefault("responder_factory", lambda _case, _gid: FakeResponder())
    kwargs.setdefault("case_dir", tmp_path)
    return TestClient(create_app(**kwargs))


def start(client: TestClient, case: Case) -> str:
    """Open the case and return the new game's id (from the redirect)."""
    response = client.post(f"/games/{case.id}", follow_redirects=False)
    assert response.status_code == 303
    return response.headers["location"].rsplit("/", 1)[-1]


@pytest.fixture
def client(tmp_path: Path, valid_case: Case) -> TestClient:
    return make_client(tmp_path, valid_case)


def test_index_lists_the_case(client: TestClient, valid_case: Case) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert valid_case.title in response.text


def test_starting_a_game_shows_the_board(client: TestClient, valid_case: Case) -> None:
    response = client.post(f"/games/{valid_case.id}", follow_redirects=True)
    assert response.status_code == 200
    assert "actions left" in response.text
    assert "Search Library" in response.text


def test_each_start_is_a_fresh_game(client: TestClient, valid_case: Case) -> None:
    first = start(client, valid_case)
    client.post(f"/games/{first}/search", data={"room_id": "library"})
    second = start(client, valid_case)
    assert first != second and first.startswith(valid_case.id)
    board = client.get(f"/games/{second}").text
    assert f"{MAX_ACTIONS} actions left" in board  # not the other player's progress


def test_index_offers_to_continue_the_last_game(client: TestClient, valid_case: Case) -> None:
    game_id = start(client, valid_case)
    response = client.get("/")
    assert "Continue your investigation" in response.text
    assert f"/games/{game_id}" in response.text


def test_searching_reveals_clues(client: TestClient, valid_case: Case) -> None:
    game_id = start(client, valid_case)
    response = client.post(f"/games/{game_id}/search", data={"room_id": "library"})
    assert response.status_code == 200
    assert "clue_ash" in response.text  # the evidence checkbox value
    assert "Evidence found" in response.text
    assert "(clue id:" not in response.text  # machine ids are not shown to people


def test_inspecting_before_searching_is_a_friendly_error(
    client: TestClient, valid_case: Case
) -> None:
    game_id = start(client, valid_case)
    response = client.post(f"/games/{game_id}/inspect", data={"clue_id": "clue_ash"})
    assert response.status_code == 200
    assert "search a room first" in response.text


def test_questioning_a_suspect(client: TestClient, valid_case: Case) -> None:
    game_id = start(client, valid_case)
    response = client.post(
        f"/games/{game_id}/question",
        data={"suspect_id": "lady_blackwood", "text": "Where were you?"},
    )
    assert response.status_code == 200
    assert "lady_blackwood: I was in the kitchen." in response.text
    assert "Where were you?" in response.text


def test_lie_detector_is_reachable_from_the_board(client: TestClient, valid_case: Case) -> None:
    game_id = start(client, valid_case)
    board = client.post(
        f"/games/{game_id}/question",
        data={"suspect_id": "lady_blackwood", "text": "Where were you?"},
    ).text
    assert f"/games/{game_id}/detector" in board
    response = client.post(f"/games/{game_id}/detector", data={"suspect_id": "lady_blackwood"})
    assert response.status_code == 200
    assert "not ready" in response.text  # no model in tests, and no charge lost


def test_accusing_ends_the_game_and_links_the_reveal(client: TestClient, valid_case: Case) -> None:
    game_id = start(client, valid_case)
    response = client.post(
        f"/games/{game_id}/accuse",
        data={"suspect_id": "butler_hobbs", "motive": "the ledger"},
    )
    assert response.status_code == 200
    assert "Correct!" in response.text
    assert f"/games/{game_id}/reveal" in response.text


def test_reveal_is_hidden_until_the_game_is_over(client: TestClient, valid_case: Case) -> None:
    game_id = start(client, valid_case)
    peek = client.get(f"/games/{game_id}/reveal", follow_redirects=False)
    assert peek.status_code == 303 and peek.headers["location"] == f"/games/{game_id}"


def test_reveal_page_shows_the_truth(client: TestClient, valid_case: Case) -> None:
    game_id = start(client, valid_case)
    client.post(f"/games/{game_id}/accuse", data={"suspect_id": "butler_hobbs"})
    response = client.get(f"/games/{game_id}/reveal")
    assert response.status_code == 200
    assert "drawing-room scene" in response.text
    assert "Hobbs" in response.text
    assert "Great Detective" in response.text


class BrokeResponder:
    def set_clues(self, suspect_id: str, clue_ids: list[str]) -> None: ...

    def answer(self, suspect_id: str, question: str) -> str:
        from alibi.llm import BudgetExceededError

        raise BudgetExceededError("daily budget reached")


def test_spend_cap_shows_a_friendly_message(tmp_path: Path, valid_case: Case) -> None:
    client = make_client(
        tmp_path, valid_case, responder_factory=lambda _case, _gid: BrokeResponder()
    )
    game_id = start(client, valid_case)
    response = client.post(
        f"/games/{game_id}/question",
        data={"suspect_id": "lady_blackwood", "text": "Where were you?"},
    )
    assert response.status_code == 200
    assert "out of budget" in response.text


def test_generating_a_case_uses_the_injected_generator(tmp_path: Path, valid_case: Case) -> None:
    client = make_client(tmp_path, valid_case, case_generator=lambda: valid_case)
    response = client.post("/case/generate", follow_redirects=True)
    assert response.status_code == 200
    assert valid_case.title in response.text


def test_generating_with_htmx_redirects_through_a_header(tmp_path: Path, valid_case: Case) -> None:
    client = make_client(tmp_path, valid_case, case_generator=lambda: valid_case)
    response = client.post("/case/generate", headers={"HX-Request": "true"})
    assert response.status_code == 204
    assert response.headers["HX-Redirect"].startswith(f"/games/{valid_case.id}-")


def test_a_failed_generation_is_a_message_not_a_500(tmp_path: Path, valid_case: Case) -> None:
    def broken() -> Case:
        raise TimeoutError("model call exceeded 300s")

    client = make_client(tmp_path, valid_case, case_generator=broken)
    response = client.post("/case/generate")
    assert response.status_code == 200
    assert "generation failed" in response.text


def test_games_survive_a_web_restart(tmp_path: Path, valid_case: Case) -> None:
    store = InMemoryGameStore()
    first = make_client(tmp_path, valid_case, game_store=store)
    game_id = start(first, valid_case)
    first.post(f"/games/{game_id}/search", data={"room_id": "library"})

    second = make_client(tmp_path, valid_case, game_store=store)  # a fresh process
    second.get(f"/games/{game_id}")
    again = make_client(tmp_path, valid_case, game_store=store).get(f"/games/{game_id}")
    assert f"{MAX_ACTIONS - 1} actions left" in again.text


# --- presentation helpers --------------------------------------------------
def make_game(case: Case) -> Game:
    return Game(case, store=InMemoryGameStore(), responder=FakeResponder())


def test_story_clock_runs_from_arrival_to_dawn(valid_case: Case) -> None:
    game = make_game(valid_case)  # death at 22:00 -> arrive 23:00, 20 x 15 min
    clock = story_clock(game)
    assert (clock.now, clock.dawn, clock.urgent) == ("23:00", "04:00", False)
    for _ in range(MAX_ACTIONS - 2):
        game.search_room("conservatory")
    later = story_clock(game)
    assert later.now == "03:30" and later.urgent


def test_clue_cards_and_friendly_text(valid_case: Case) -> None:
    game = make_game(valid_case)
    result = game.search_room("library")
    cards = clue_cards(game)
    assert {card["room"] for card in cards} == {"Library"}
    assert "clue id" in result.message and "clue id" not in friendly(result.message)


def test_rank_and_review(valid_case: Case) -> None:
    game = make_game(valid_case)
    game.search_room("library")
    game.question("butler_hobbs", "Explain this.", ["clue_ash", "clue_cufflink"])
    game.accuse("butler_hobbs")
    assert detective_rank(game)["rank"] == "Great Detective"
    review = case_review(game)
    assert review["genuine_found"] == review["genuine_total"] == 2
    assert review["shown_to_murderer"] == 2 and review["missed"] == []

    wrong = make_game(valid_case)
    wrong.accuse("dr_penrose")
    assert detective_rank(wrong)["rank"] == "Sergeant"
    assert len(case_review(wrong)["missed"]) == 2


# --- assets ----------------------------------------------------------------
HTMX_SHA384 = "0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq"  # htmx.org 2.0.3


def test_htmx_is_served_locally_and_untampered(client: TestClient) -> None:
    import base64
    import hashlib

    page = client.get("/").text
    assert '<script src="/static/htmx.min.js"' in page
    response = client.get("/static/htmx.min.js")
    assert response.status_code == 200
    digest = base64.b64encode(hashlib.sha384(response.content).digest()).decode()
    assert digest == HTMX_SHA384


def test_templates_load_no_third_party_scripts() -> None:
    from alibi.web import TEMPLATES_DIR

    for template in TEMPLATES_DIR.rglob("*.html"):
        text = template.read_text(encoding="utf-8")
        assert "unpkg.com" not in text and "googleapis" not in text, template.name
