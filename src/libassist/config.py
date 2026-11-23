"""Configuration, read from environment variables (12-factor style).

Everything has a sensible default so `make test` and `make demo` work with no setup.
Only the interactive chat and the eval suite need an LLM endpoint (see .env.example).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# src/libassist/config.py -> parents[2] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Settings for the REST API process."""

    db_path: Path
    seed_path: Path
    policies_dir: Path
    today_override: date | None = None
    # Optional real embedding model (any OpenAI-compatible /embeddings endpoint, e.g. Ollama).
    embeddings_base_url: str | None = None
    embeddings_model: str = "nomic-embed-text"
    embeddings_api_key: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        today = _env("LIBASSIST_TODAY")
        return cls(
            db_path=Path(_env("LIBASSIST_DB_PATH", str(REPO_ROOT / "libassist.db"))),
            seed_path=Path(_env("LIBASSIST_SEED_PATH", str(REPO_ROOT / "data" / "seed.json"))),
            policies_dir=Path(_env("LIBASSIST_POLICIES_DIR", str(REPO_ROOT / "data" / "policies"))),
            today_override=date.fromisoformat(today) if today else None,
            embeddings_base_url=_env("EMBEDDINGS_BASE_URL"),
            embeddings_model=_env("EMBEDDINGS_MODEL", "nomic-embed-text"),
            embeddings_api_key=_env("EMBEDDINGS_API_KEY"),
        )

    def today(self) -> date:
        """The business date. Overridable so tests, demos and evals are deterministic."""
        return self.today_override or date.today()


def api_base_url() -> str:
    """Where the MCP server finds the REST API."""
    return _env("LIBASSIST_API_URL", "http://127.0.0.1:8000")
