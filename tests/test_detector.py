from __future__ import annotations

from pathlib import Path

from alibi import config
from alibi import detector as detector_module
from alibi.case import Case
from alibi.detector import LieDetector, Prediction
from alibi.game import Game, InMemoryGameStore


class FakeResponder:
    def set_clues(self, suspect_id: str, clue_ids: list[str]) -> None: ...

    def answer(self, suspect_id: str, question: str) -> str:
        return "I was in the kitchen."


class FakeDetector:
    def __init__(self, label: str = "FALSE") -> None:
        self.label = label
        self.seen: list[str] = []

    def predict(self, statement: str) -> Prediction:
        self.seen.append(statement)
        probability = 0.9 if self.label == "FALSE" else 0.1
        return Prediction(label=self.label, probability=probability)


def make_game(case: Case, detector=None) -> Game:
    return Game(case, store=InMemoryGameStore(), responder=FakeResponder(), detector=detector)


def test_available_is_false_without_files(tmp_path: Path) -> None:
    assert LieDetector.available(tmp_path) is False


def test_from_settings_is_none_without_a_model(tmp_path: Path, monkeypatch) -> None:
    settings = config.Settings(detector_dir=tmp_path)
    monkeypatch.setattr(detector_module, "get_settings", lambda: settings)
    assert LieDetector.from_settings() is None


def test_detector_flags_a_lie(valid_case: Case) -> None:
    fake = FakeDetector("FALSE")
    game = make_game(valid_case, detector=fake)
    game.question("lady_blackwood", "Where were you?")

    result = game.lie_detector()

    assert result.ok and "trembles" in result.message
    assert result.data["label"] == "FALSE"
    assert fake.seen == ["I was in the kitchen."]


def test_detector_reads_a_true_answer(valid_case: Case) -> None:
    game = make_game(valid_case, detector=FakeDetector("TRUE"))
    game.question("lady_blackwood", "Where were you?")
    assert "holds steady" in game.lie_detector().message


def test_detector_without_an_answer_is_a_friendly_error(valid_case: Case) -> None:
    game = make_game(valid_case, detector=FakeDetector())
    result = game.lie_detector()
    assert not result.ok and "nothing to test" in result.message


def test_detector_is_a_stub_when_no_model_is_installed(valid_case: Case) -> None:
    game = make_game(valid_case, detector=None)
    result = game.lie_detector()
    assert result.ok and "not ready" in result.message
