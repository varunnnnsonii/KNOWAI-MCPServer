"""Data models for changelog entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChangelogEntry:
    """A single parsed changelog entry with structured metadata.

    Each entry corresponds to one row in the changelog table,
    preserving the What/Why/Where/When schema from the source.
    """

    entry_number: int
    category: str  # e.g. "Fix", "Perf", "Security", "Refactor", "Migration"
    title: str  # Extracted title from the "What" column
    description: str  # Full "What" column text
    rationale: str  # "Why" column
    affected_paths: list[str] = field(default_factory=list)  # Parsed from "Where"
    module: str = ""  # Extracted module name (e.g. "auth", "fleet")
    date: datetime | None = None  # Parsed date
    date_str: str = ""  # Original date string
    raw_text: str = ""  # Full row as text
    source_file: str = ""  # Which file this came from
    section: str = "changelog"  # "changelog" | "key_decision" | "quick_reference"

    @property
    def chunk_id(self) -> str:
        """Deterministic, idempotent ID for this chunk."""
        return f"{self.source_file}::{self.section}::{self.entry_number}"

    @property
    def embedding_text(self) -> str:
        """Construct the text that will be embedded for semantic search.

        Combines the most semantically relevant fields into a single
        string optimized for retrieval. Metadata (module, date) is
        included to improve filtering by similarity.
        """
        parts = []
        if self.category:
            parts.append(f"[{self.category}]")
        parts.append(self.title)
        if self.description and self.description != self.title:
            parts.append(f"— {self.description}")
        if self.rationale:
            parts.append(f"Rationale: {self.rationale}")
        if self.module:
            parts.append(f"Module: {self.module}")
        if self.date_str:
            parts.append(f"Date: {self.date_str}")
        return " ".join(parts)

    def to_metadata(self) -> dict:
        """Convert to a flat metadata dict suitable for ChromaDB storage."""
        return {
            "entry_number": self.entry_number,
            "category": self.category,
            "title": self.title,
            "rationale": self.rationale,
            "module": self.module,
            "date_str": self.date_str,
            "date_iso": self.date.isoformat() if self.date else "",
            "affected_paths": "|".join(self.affected_paths),
            "source_file": self.source_file,
            "section": self.section,
        }

    def to_result(self, relevance_score: float = 0.0) -> dict:
        """Convert to the output format returned by MCP tools."""
        return {
            "entry_number": self.entry_number,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "category": self.category,
            "module": self.module,
            "affected_paths": self.affected_paths,
            "date": self.date_str,
            "relevance_score": round(relevance_score, 4),
            "source_file": self.source_file,
            "section": self.section,
        }
