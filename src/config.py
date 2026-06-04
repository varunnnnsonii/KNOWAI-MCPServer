"""Configuration management.

All settings are loaded from environment variables with sensible defaults.
Uses pydantic-settings for typed, validated configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Paths ---
    changelog_dir: Path = Path("./data/changelogs")
    data_dir: Path = Path("./data")

    # --- Embedding Model ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Search ---
    search_default_limit: int = 5
    search_max_limit: int = 20
    relevance_threshold: float = 0.25

    # --- Logging ---
    log_level: str = "INFO"

    # --- Server ---
    server_name: str = "changelog-semantic-search"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Derived Paths ---
    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25_index.pkl"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "file_manifest.json"

    @property
    def index_metadata_path(self) -> Path:
        return self.data_dir / "index_metadata.json"

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.changelog_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Factory that returns a validated Settings instance."""
    return Settings()
