from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alibi.case import Case
from alibi.game import InMemoryGameStore
from alibi.web import create_app


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


def test_searching_reveals_clues(client: TestClient, valid_case: Case) -> None:
    client.post(f"/games/{valid_case.id}")
    response = client.post(f"/games/{valid_case.id}/search", data={"room_id": "library"})
    assert response.status_code == 200
    assert "clue_ash" in response.text
    assert "Evidence found" in response.text


def test_inspecting_before_searching_is_a_friendly_error(
    client: TestClient, valid_case: Case
) -> None:
    client.post(f"/games/{valid_case.id}")
    response = client.post(f"/games/{valid_case.id}/inspect", data={"clue_id": "clue_ash"})
    assert response.status_code == 200
    assert "search a room first" in response.text


def test_questioning_a_suspect(client: TestClient, valid_case: Case) -> None:
    client.post(f"/games/{valid_case.id}")
    response = client.post(
        f"/games/{valid_case.id}/question",
        data={"suspect_id": "lady_blackwood", "text": "Where were you?"},
    )
    assert response.status_code == 200
    assert "lady_blackwood: I was in the kitchen." in response.text
    assert "Where were you?" in response.text


def test_accusing_ends_the_game_and_links_the_reveal(client: TestClient, valid_case: Case) -> None:
    client.post(f"/games/{valid_case.id}")
    response = client.post(
        f"/games/{valid_case.id}/accuse",
        data={"suspect_id": "butler_hobbs", "motive": "the ledger"},
    )
    assert response.status_code == 200
    assert "Correct!" in response.text
    assert f"/games/{valid_case.id}/reveal" in response.text


def test_reveal_page_shows_the_truth(client: TestClient, valid_case: Case) -> None:
    client.post(f"/games/{valid_case.id}")
    client.post(f"/games/{valid_case.id}/accuse", data={"suspect_id": "butler_hobbs"})
    response = client.get(f"/games/{valid_case.id}/reveal")
    assert response.status_code == 200
    assert "What really happened" in response.text
    assert "Hobbs" in response.text


class BrokeResponder:
    def set_clues(self, suspect_id: str, clue_ids: list[str]) -> None: ...

    def answer(self, suspect_id: str, question: str) -> str:
        from alibi.llm import BudgetExceededError

        raise BudgetExceededError("daily budget reached")


def test_spend_cap_shows_a_friendly_message(tmp_path: Path, valid_case: Case) -> None:
    client = make_client(
        tmp_path, valid_case, responder_factory=lambda _case, _gid: BrokeResponder()
    )
    client.post(f"/games/{valid_case.id}")
    response = client.post(
        f"/games/{valid_case.id}/question",
        data={"suspect_id": "lady_blackwood", "text": "Where were you?"},
    )
    assert response.status_code == 200
    assert "out of budget" in response.text


def test_serves_htmx_locally(client: TestClient) -> None:
    response = client.get("/static/htmx.min.js")
    assert response.status_code == 200
    assert "htmx" in response.text


def test_generating_a_case_uses_the_injected_generator(tmp_path: Path, valid_case: Case) -> None:
    client = make_client(tmp_path, valid_case, case_generator=lambda: valid_case)
    response = client.post("/case/generate", follow_redirects=True)
    assert response.status_code == 200
    assert valid_case.title in response.text
