"""File change tracker for incremental indexing.

Tracks changelog files by SHA-256 hash to enable:
  - Skip unchanged files on re-index
  - Detect modified files for re-processing
  - Detect deleted files for cleanup
  - Atomic manifest persistence (write-tmp-then-rename)
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.observability import get_logger

logger = get_logger(__name__)


class FileManifest:
    """Persistent manifest of tracked changelog files.

    Each entry stores:
        - file_hash: SHA-256 of the file contents
        - last_indexed: ISO timestamp of the last successful index
        - entry_count: Number of entries extracted from the file

    The manifest is stored as a JSON file on disk with atomic writes.
    """

    def __init__(self, manifest_path: Path):
        self._path = manifest_path
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the manifest from disk, or start empty."""
        if not self._path.exists():
            self._entries = {}
            return

        try:
            raw = self._path.read_text(encoding="utf-8")
            self._entries = json.loads(raw)
            logger.info(
                "File manifest loaded",
                extra={"count": len(self._entries), "file": str(self._path)},
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Corrupt manifest, starting fresh",
                extra={"error": str(e), "file": str(self._path)},
            )
            self._entries = {}

    def save(self) -> None:
        """Atomically save the manifest to disk.

        Writes to a temporary file first, then renames to avoid
        corruption if the process crashes mid-write.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write to temp file in the same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent),
                prefix=".manifest_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._entries, f, indent=2, sort_keys=True)

                # Atomic rename
                os.replace(tmp_path, str(self._path))
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            logger.info(
                "File manifest saved",
                extra={"count": len(self._entries), "file": str(self._path)},
            )
        except Exception as e:
            logger.error(
                "Failed to save manifest",
                extra={"error": str(e), "file": str(self._path)},
            )
            raise

    # ------------------------------------------------------------------
    # File tracking
    # ------------------------------------------------------------------

    def get_entry(self, filepath: str) -> dict[str, Any] | None:
        """Get the manifest entry for a file, or None if not tracked."""
        return self._entries.get(filepath)

    def update_entry(
        self,
        filepath: str,
        file_hash: str,
        entry_count: int,
    ) -> None:
        """Update or create a manifest entry for a file."""
        self._entries[filepath] = {
            "file_hash": file_hash,
            "last_indexed": datetime.now(timezone.utc).isoformat(),
            "entry_count": entry_count,
        }

    def remove_entry(self, filepath: str) -> None:
        """Remove a file from the manifest."""
        self._entries.pop(filepath, None)

    def get_all_tracked_files(self) -> set[str]:
        """Return the set of all tracked file paths."""
        return set(self._entries.keys())

    @property
    def total_entries(self) -> int:
        """Total number of changelog entries across all tracked files."""
        return sum(e.get("entry_count", 0) for e in self._entries.values())


# ------------------------------------------------------------------
# Hashing
# ------------------------------------------------------------------

def compute_file_hash(filepath: Path) -> str:
    """Compute the SHA-256 hash of a file's contents.

    Args:
        filepath: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ------------------------------------------------------------------
# Change detection
# ------------------------------------------------------------------

def detect_changes(
    manifest: FileManifest,
    current_files: list[Path],
) -> tuple[list[Path], list[Path], list[str]]:
    """Compare current files against the manifest to detect changes.

    Args:
        manifest: The persisted file manifest.
        current_files: List of currently discovered changelog files.

    Returns:
        Tuple of (new_or_changed, unchanged, deleted):
        - new_or_changed: Files that are new or have changed since last index.
        - unchanged: Files that have not changed.
        - deleted: File paths in manifest but no longer on disk.
    """
    new_or_changed: list[Path] = []
    unchanged: list[Path] = []

    current_paths = {str(f) for f in current_files}
    tracked_paths = manifest.get_all_tracked_files()

    for filepath in current_files:
        str_path = str(filepath)
        entry = manifest.get_entry(str_path)

        if entry is None:
            # New file
            new_or_changed.append(filepath)
            continue

        current_hash = compute_file_hash(filepath)
        if current_hash != entry.get("file_hash"):
            # File changed
            new_or_changed.append(filepath)
        else:
            # Unchanged
            unchanged.append(filepath)

    # Detect deleted files
    deleted = [p for p in tracked_paths if p not in current_paths]

    logger.info(
        "Change detection complete",
        extra={
            "files_processed": len(current_files),
            "count": len(new_or_changed),  # new/changed count
            "unchanged": len(unchanged),
            "deleted": len(deleted),
        },
    )
    return new_or_changed, unchanged, deleted
