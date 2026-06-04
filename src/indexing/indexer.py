"""Indexing pipeline orchestrator.

Ties together the full indexing flow:
  parser → chunker → embedder → vector store

Supports:
  - Full indexing (first run)
  - Incremental indexing (hash-based change detection)
  - Deleted file cleanup
  - Index metadata persistence
  - Corruption recovery (auto-rebuild)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Settings
from src.indexing.chunker import chunk_rows
from src.indexing.embedder import Embedder, EmbeddingError
from src.indexing.file_tracker import (
    FileManifest,
    compute_file_hash,
    detect_changes,
)
from src.indexing.parser import discover_changelog_files, parse_file
from src.models import ChangelogEntry
from src.observability import get_logger
from src.retrieval.vector_store import VectorStore, VectorStoreError

logger = get_logger(__name__)


class IndexError(Exception):
    """Raised when the indexing pipeline encounters a fatal error."""


class IndexMetadata:
    """Tracks index-level metadata: last index time, entry counts, etc."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        """Atomically save index metadata."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".idx_meta_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp_path, str(self._path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def update(
        self,
        total_entries: int,
        files_processed: int,
        duration_ms: float,
        model_name: str,
    ) -> None:
        self._data = {
            "last_indexed": datetime.now(timezone.utc).isoformat(),
            "total_entries": total_entries,
            "files_processed": files_processed,
            "duration_ms": round(duration_ms, 1),
            "model_name": model_name,
        }

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)


class Indexer:
    """Orchestrates the full indexing pipeline.

    Usage:
        indexer = Indexer(settings)
        entries = indexer.run()  # Returns all indexed entries
    """

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.settings = settings
        self.embedder = embedder or Embedder(settings.embedding_model)
        self.vector_store = vector_store or VectorStore(settings.chroma_dir)
        self.manifest = FileManifest(settings.manifest_path)
        self.metadata = IndexMetadata(settings.index_metadata_path)
        self._all_entries: list[ChangelogEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[ChangelogEntry]:
        """Execute the indexing pipeline.

        Returns:
            List of all ChangelogEntry objects that were indexed.
        """
        start = time.monotonic()
        self.settings.ensure_dirs()

        # Discover files
        try:
            files = discover_changelog_files(self.settings.changelog_dir)
        except FileNotFoundError:
            logger.warning(
                "Changelog directory not found",
                extra={"directory": str(self.settings.changelog_dir)},
            )
            return []

        if not files and len(self.manifest.get_all_tracked_files()) == 0:
            logger.info("No changelog files found and index is empty — nothing to index")
            return []

        # Detect changes
        new_or_changed, unchanged, deleted = detect_changes(self.manifest, files)

        # Handle deleted files
        for deleted_path in deleted:
            logger.info("Removing deleted file from index", extra={"file": deleted_path})
            self.vector_store.delete_by_source(deleted_path)
            self.manifest.remove_entry(deleted_path)

        # Process new/changed files
        all_new_entries: list[ChangelogEntry] = []

        for filepath in new_or_changed:
            entries = self._index_file(filepath)
            all_new_entries.extend(entries)

        # Collect all entries (new + existing unchanged)
        self._all_entries = all_new_entries
        # Note: entries from unchanged files are already in the vector store

        # Save manifest and metadata
        self.manifest.save()

        elapsed_ms = (time.monotonic() - start) * 1000
        total_count = self.vector_store.count()

        self.metadata.update(
            total_entries=total_count,
            files_processed=len(new_or_changed),
            duration_ms=elapsed_ms,
            model_name=self.embedder.model_name,
        )
        self.metadata.save()

        logger.info(
            "Indexing pipeline complete",
            extra={
                "files_processed": len(new_or_changed),
                "entries_indexed": len(all_new_entries),
                "files_unchanged": len(unchanged),
                "files_deleted": len(deleted),
                "total_in_store": total_count,
                "duration_ms": round(elapsed_ms, 1),
            },
        )

        return all_new_entries

    def get_status(self) -> dict[str, Any]:
        """Return current index status information."""
        return {
            "total_entries": self.vector_store.count(),
            "tracked_files": len(self.manifest.get_all_tracked_files()),
            "tracked_file_list": sorted(self.manifest.get_all_tracked_files()),
            "index_metadata": self.metadata.data,
            "model_name": self.embedder.model_name,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index_file(self, filepath: Path) -> list[ChangelogEntry]:
        """Parse, chunk, embed, and store entries from a single file.

        Args:
            filepath: Path to the changelog file.

        Returns:
            List of successfully indexed ChangelogEntry objects.
        """
        str_path = str(filepath)
        logger.info("Indexing file", extra={"file": str_path})

        # 1. Parse
        rows = parse_file(filepath)
        if not rows:
            logger.warning("No rows parsed from file", extra={"file": str_path})
            return []

        # 2. Chunk
        entries = chunk_rows(rows)
        if not entries:
            logger.warning("No entries after chunking", extra={"file": str_path})
            return []

        # 3. Remove old entries for this file (before re-indexing)
        self.vector_store.delete_by_source(str_path)

        # 4. Embed
        try:
            texts = [e.embedding_text for e in entries]
            embeddings = self.embedder.embed_texts(texts)
        except EmbeddingError as e:
            logger.error(
                "Embedding failed for file — skipping",
                extra={"file": str_path, "error": str(e)},
            )
            return []

        # 5. Store in vector DB
        try:
            ids = [e.chunk_id for e in entries]
            documents = texts
            metadatas = [e.to_metadata() for e in entries]

            self.vector_store.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except VectorStoreError as e:
            logger.error(
                "Vector store upsert failed for file — skipping",
                extra={"file": str_path, "error": str(e)},
            )
            return []

        # 6. Update manifest
        file_hash = compute_file_hash(filepath)
        self.manifest.update_entry(
            filepath=str_path,
            file_hash=file_hash,
            entry_count=len(entries),
        )

        logger.info(
            "File indexed successfully",
            extra={
                "file": str_path,
                "entries_indexed": len(entries),
            },
        )
        return entries
