from __future__ import annotations

from pathlib import Path

import pytest

from alibi import config


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALIBI_MODEL", raising=False)
    for role in [*config.ROLE_NAMES, "embedding"]:
        monkeypatch.delenv(f"ALIBI_MODEL_{role.upper()}", raising=False)


def test_defaults_are_used_without_overrides() -> None:
    models = config._load_models(Path("does/not/exist.yaml"))
    assert models == config.DEFAULT_MODELS


def test_yaml_values_load(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("generator: someone/big-model\n")
    models = config._load_models(path)
    assert models["generator"] == "someone/big-model"
    assert models["suspect"] == config.DEFAULT_MODELS["suspect"]


def test_alibi_model_sets_every_chat_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIBI_MODEL", "openrouter/deepseek/deepseek-v4.1-flash")
    models = config._load_models(Path("does/not/exist.yaml"))

    for role in config.ROLE_NAMES:
        assert models[role] == "openrouter/deepseek/deepseek-v4.1-flash"
    # The embedding model is deliberately left alone.
    assert models["embedding"] == config.DEFAULT_MODELS["embedding"]


def test_per_role_override_beats_alibi_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIBI_MODEL", "openrouter/deepseek/deepseek-v4.1-flash")
    monkeypatch.setenv("ALIBI_MODEL_GENERATOR", "openrouter/moonshotai/kimi")
    models = config._load_models(Path("does/not/exist.yaml"))

    assert models["generator"] == "openrouter/moonshotai/kimi"
    assert models["suspect"] == "openrouter/deepseek/deepseek-v4.1-flash"
