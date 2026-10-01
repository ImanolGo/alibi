"""Settings: environment variables + ``config/models.yaml``.

Nothing here talks to a model. ``llm.py`` reads the resolved settings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

Role = Literal["generator", "suspect", "judge", "solver"]
ROLE_NAMES: tuple[str, ...] = get_args(Role)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS: dict[str, str] = {
    "generator": "openai/gpt-4o",
    "suspect": "openai/gpt-4o-mini",
    "judge": "openai/gpt-4o-mini",
    "solver": "openai/gpt-4o-mini",
    "embedding": "openrouter/openai/text-embedding-3-small",
}

# Dimension of the embedding vectors stored in Postgres (pgvector).
EMBEDDING_DIM = 1536


class Settings(BaseModel):
    """Resolved runtime configuration."""

    daily_budget_usd: float = Field(default=2.0, ge=0)
    models: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODELS))
    embedding_dim: int = EMBEDDING_DIM
    request_timeout_s: float = Field(default=180.0, gt=0)
    generator_timeout_s: float = Field(default=300.0, gt=0)
    database_url: str = "postgresql+psycopg://alibi:alibi@localhost:5432/alibi"
    project_root: Path = PROJECT_ROOT
    cases_dir: Path = PROJECT_ROOT / "cases"
    setting_path: Path = PROJECT_ROOT / "config" / "setting.yaml"
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"

    @property
    def embedding_model(self) -> str:
        return self.models.get("embedding", DEFAULT_MODELS["embedding"])


def _load_models(path: Path) -> dict[str, str]:
    models = dict(DEFAULT_MODELS)
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        for key, value in data.items():
            if isinstance(value, str):
                models[str(key)] = value
    # ``ALIBI_MODEL`` sets one model for every chat role (never embedding).
    default = os.environ.get("ALIBI_MODEL")
    if default:
        for role in ROLE_NAMES:
            models[role] = default
    # A per-role override wins over both the file and ``ALIBI_MODEL``.
    for role in list(models):
        override = os.environ.get(f"ALIBI_MODEL_{role.upper()}")
        if override:
            models[role] = override
    return models


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once. Call ``get_settings.cache_clear()`` in tests."""
    load_dotenv(PROJECT_ROOT / ".env")
    root = Path(os.environ.get("ALIBI_ROOT", str(PROJECT_ROOT)))
    budget = float(os.environ.get("ALIBI_DAILY_BUDGET_USD", "2.0"))
    timeout = float(os.environ.get("ALIBI_REQUEST_TIMEOUT_S", "180"))
    generator_timeout = float(os.environ.get("ALIBI_GENERATOR_TIMEOUT_S", "300"))
    return Settings(
        daily_budget_usd=budget,
        models=_load_models(root / "config" / "models.yaml"),
        request_timeout_s=timeout,
        generator_timeout_s=generator_timeout,
        database_url=os.environ.get(
            "ALIBI_DATABASE_URL",
            "postgresql+psycopg://alibi:alibi@localhost:5432/alibi",
        ),
        project_root=root,
        cases_dir=Path(os.environ.get("ALIBI_CASES_DIR", str(root / "cases"))),
        setting_path=Path(os.environ.get("ALIBI_SETTING", str(root / "config" / "setting.yaml"))),
        phoenix_endpoint=os.environ.get(
            "ALIBI_PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces"
        ),
    )


def load_setting(path: Path | None = None) -> dict[str, Any]:
    """Read the setting description handed to the case generator."""
    path = path or get_settings().setting_path
    return yaml.safe_load(path.read_text()) or {}
