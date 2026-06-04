"""Markdown changelog parser.

Parses changelog files in Markdown format into raw row data.
Handles:
  - The main changelog table (## Changelog section)
  - The key decisions table (## Key decisions section)
  - Malformed rows gracefully (skip + log warning)
  - Multiple files in a directory
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

from src.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Section detection patterns
# ---------------------------------------------------------------------------
_CHANGELOG_HEADER_RE = re.compile(r"^##\s+.*Changelog", re.IGNORECASE)
_DECISIONS_HEADER_RE = re.compile(r"^##\s+.*Key\s+decisions", re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|$")
_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$")

# Category prefixes commonly found in the "What" column
CATEGORY_PREFIXES = [
    "Fix", "Perf", "Security", "Refactor", "Migration", "Tests",
    "Worker", "Deprecate", "Observability", "Chore", "DX", "Deploy",
    "Admin", "Realtime", "Frontend",
]

# Module extraction: look for paths like apps/api/src/modules/<name>/
_MODULE_RE = re.compile(r"modules/(\w+)/")
# Also catch top-level dirs like apps/api/src/auth/
_AUTH_MODULE_RE = re.compile(r"src/(auth|core|realtime)/")
# Package paths like packages/db/ or packages/ui/
_PACKAGE_RE = re.compile(r"packages/(\w[\w-]*)/")


def discover_changelog_files(directory: Path) -> list[Path]:
    """Find all Markdown files in the given directory.

    Args:
        directory: Path to scan for .md files.

    Returns:
        Sorted list of Path objects for all Markdown files found.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Changelog directory not found: {directory}")

    md_files = sorted(directory.glob("*.md"))
    logger.info(
        "Discovered changelog files",
        extra={"count": len(md_files), "directory": str(directory)},
    )
    return md_files


# ---------------------------------------------------------------------------
# Row dataclass (raw parsed data, before semantic enrichment)
# ---------------------------------------------------------------------------
class RawRow:
    """A raw parsed row from a Markdown table."""

    __slots__ = ("cells", "section", "source_file", "line_number")

    def __init__(
        self,
        cells: list[str],
        section: str,
        source_file: str,
        line_number: int,
    ):
        self.cells = cells
        self.section = section
        self.source_file = source_file
        self.line_number = line_number


def parse_file(filepath: Path) -> list[RawRow]:
    """Parse a single Markdown file into raw table rows.

    Identifies the Changelog and Key Decisions sections and extracts
    each table row as a list of cell strings.

    Args:
        filepath: Path to the Markdown file.

    Returns:
        List of RawRow objects with cell data and section metadata.
    """
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(
            "Failed to read file",
            extra={"file": str(filepath), "error": str(e)},
        )
        return []

    lines = text.splitlines()
    rows: list[RawRow] = []
    current_section: str | None = None
    in_table = False
    header_seen = False

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Detect section headers
        if _CHANGELOG_HEADER_RE.match(stripped):
            current_section = "changelog"
            in_table = False
            header_seen = False
            continue
        elif _DECISIONS_HEADER_RE.match(stripped):
            current_section = "key_decision"
            in_table = False
            header_seen = False
            continue
        elif stripped.startswith("## "):
            # Any other H2 resets section tracking
            current_section = None
            in_table = False
            header_seen = False
            continue

        if current_section is None:
            continue

        # Table row handling
        match = _TABLE_ROW_RE.match(stripped)
        if match:
            if _SEPARATOR_RE.match(stripped):
                # This is the |---|---| separator line
                in_table = True
                continue

            if not in_table:
                # This is the header row (before separator)
                header_seen = True
                continue

            # Data row — split cells
            raw_cells = [c.strip() for c in match.group(1).split("|")]
            if len(raw_cells) >= 2:  # Need at least 2 cells for useful data
                rows.append(
                    RawRow(
                        cells=raw_cells,
                        section=current_section,
                        source_file=str(filepath),
                        line_number=line_num,
                    )
                )
        else:
            # Non-table line inside a section — end of table
            if in_table and stripped and not stripped.startswith(">"):
                in_table = False

    logger.info(
        "Parsed file",
        extra={
            "file": str(filepath),
            "total_rows": len(rows),
            "changelog_rows": sum(1 for r in rows if r.section == "changelog"),
            "decision_rows": sum(1 for r in rows if r.section == "key_decision"),
        },
    )
    return rows


def extract_category(what_text: str) -> str:
    """Extract the category prefix from the What column.

    Examples:
        "**Fix: Geofence duplicate** ..." → "Fix"
        "**Perf: cache reads** ..."       → "Perf"
        "**Driver schema** ..."           → "Feature"

    Args:
        what_text: Raw text from the What column.

    Returns:
        Category string, or "Feature" if no known prefix found.
    """
    # Strip markdown bold markers
    clean = re.sub(r"\*\*", "", what_text).strip()

    for prefix in CATEGORY_PREFIXES:
        if clean.lower().startswith(prefix.lower() + ":"):
            return prefix
        if clean.lower().startswith(prefix.lower() + " "):
            return prefix

    return "Feature"


def extract_title(what_text: str) -> str:
    """Extract a clean title from the What column.

    Removes markdown bold markers and extracts the primary title
    (before the em-dash description).

    Examples:
        "**Fix: Geofence duplicate on retry** — missing..." → "Fix: Geofence duplicate on retry"
        "**Driver schema** — `drivers`..." → "Driver schema"

    Args:
        what_text: Raw text from the What column.

    Returns:
        Clean title string.
    """
    # Extract text inside bold markers first
    bold_match = re.search(r"\*\*(.+?)\*\*", what_text)
    if bold_match:
        return bold_match.group(1).strip()

    # Fallback: take text before em-dash
    if " — " in what_text:
        return what_text.split(" — ")[0].strip()

    return what_text.strip()[:200]  # Safety cap


def extract_module(where_text: str) -> str:
    """Extract the primary module name from the Where column.

    Looks for patterns like:
        apps/api/src/modules/<module_name>/
        apps/api/src/auth/
        packages/<package_name>/

    Args:
        where_text: Raw text from the Where column.

    Returns:
        Module name string, or empty string if not found.
    """
    # Try module paths first
    match = _MODULE_RE.search(where_text)
    if match:
        return match.group(1)

    # Try auth/core/realtime
    match = _AUTH_MODULE_RE.search(where_text)
    if match:
        return match.group(1)

    # Try packages
    match = _PACKAGE_RE.search(where_text)
    if match:
        return match.group(1)

    return ""


def extract_paths(where_text: str) -> list[str]:
    """Extract file/directory paths from the Where column.

    Handles backtick-wrapped paths and comma-separated lists.

    Args:
        where_text: Raw text from the Where column.

    Returns:
        List of path strings.
    """
    # Find all backtick-wrapped paths
    paths = re.findall(r"`([^`]+)`", where_text)
    if paths:
        return paths

    # Fallback: split by comma if no backtick paths
    if where_text.strip():
        return [p.strip() for p in where_text.split(",") if p.strip()]

    return []
