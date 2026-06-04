"""Chunker — converts raw parsed rows into ChangelogEntry objects.

Handles:
  - Changelog table rows (5-column: #, What, Why, Where, When)
  - Key Decisions rows (3-column: Decision, Rationale, When)
  - Date parsing across multiple formats
  - Graceful handling of malformed rows
"""

from __future__ import annotations

import re
from datetime import datetime

from dateutil import parser as dateutil_parser

from src.indexing.parser import RawRow, extract_category, extract_title, extract_module, extract_paths
from src.models import ChangelogEntry
from src.observability import get_logger

logger = get_logger(__name__)


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string flexibly, returning None on failure.

    Handles formats like:
        "Jun 3, 2026 09:43 IST"
        "Jan 2024"
        "2024"
        "Sep 21, 2025 10:01 IST"

    Args:
        date_str: Raw date string from the changelog.

    Returns:
        Parsed datetime or None if unparseable.
    """
    if not date_str or not date_str.strip():
        return None

    clean = date_str.strip()
    # Remove timezone abbreviations that dateutil doesn't know
    clean = re.sub(r"\bIST\b", "", clean).strip()
    clean = re.sub(r"\bPST\b", "", clean).strip()
    clean = re.sub(r"\bEST\b", "", clean).strip()
    clean = re.sub(r"\bUTC\b", "", clean).strip()

    try:
        return dateutil_parser.parse(clean, fuzzy=True)
    except (ValueError, OverflowError):
        return None


def _parse_entry_number(cell: str) -> int:
    """Extract an integer entry number from a cell, defaulting to 0."""
    try:
        return int(cell.strip())
    except (ValueError, TypeError):
        return 0


def chunk_changelog_row(row: RawRow) -> ChangelogEntry | None:
    """Convert a raw changelog table row into a ChangelogEntry.

    Expected cell format: [#, What, Why, Where, When]

    Args:
        row: RawRow with cells from the changelog section.

    Returns:
        ChangelogEntry or None if the row is malformed.
    """
    cells = row.cells
    if len(cells) < 4:
        logger.warning(
            "Skipping malformed changelog row (too few cells)",
            extra={"file": row.source_file, "line_number": row.line_number, "cells": len(cells)},
        )
        return None

    # Cells: [#, What, Why, Where, When] — but some might have extra pipes
    entry_num = _parse_entry_number(cells[0])
    what_text = cells[1].strip() if len(cells) > 1 else ""
    why_text = cells[2].strip() if len(cells) > 2 else ""
    where_text = cells[3].strip() if len(cells) > 3 else ""
    when_text = cells[4].strip() if len(cells) > 4 else ""

    if not what_text:
        return None

    title = extract_title(what_text)
    category = extract_category(what_text)
    module = extract_module(where_text)
    paths = extract_paths(where_text)
    date = _parse_date(when_text)

    # Build the full description (what column minus bold markers)
    description = re.sub(r"\*\*", "", what_text).strip()

    return ChangelogEntry(
        entry_number=entry_num,
        category=category,
        title=title,
        description=description,
        rationale=why_text,
        affected_paths=paths,
        module=module,
        date=date,
        date_str=when_text,
        raw_text=f"#{entry_num} | {what_text} | {why_text} | {where_text} | {when_text}",
        source_file=row.source_file,
        section=row.section,
    )


def chunk_decision_row(row: RawRow, index: int) -> ChangelogEntry | None:
    """Convert a raw key-decisions table row into a ChangelogEntry.

    Expected cell format: [Decision, Rationale, When]

    Args:
        row: RawRow with cells from the key_decision section.
        index: Sequential index for this decision (used as entry_number).

    Returns:
        ChangelogEntry or None if the row is malformed.
    """
    cells = row.cells
    if len(cells) < 2:
        logger.warning(
            "Skipping malformed decision row (too few cells)",
            extra={"file": row.source_file, "line_number": row.line_number, "cells": len(cells)},
        )
        return None

    decision_text = cells[0].strip()
    rationale_text = cells[1].strip() if len(cells) > 1 else ""
    when_text = cells[2].strip() if len(cells) > 2 else ""

    if not decision_text:
        return None

    title = extract_title(decision_text)
    # Clean markdown bold from full description
    description = re.sub(r"\*\*", "", decision_text).strip()
    date = _parse_date(when_text)

    return ChangelogEntry(
        entry_number=10000 + index,  # Offset to avoid collision with changelog entries
        category="Decision",
        title=title,
        description=description,
        rationale=rationale_text,
        affected_paths=[],
        module="",
        date=date,
        date_str=when_text,
        raw_text=f"Decision: {decision_text} | {rationale_text} | {when_text}",
        source_file=row.source_file,
        section=row.section,
    )


def chunk_rows(rows: list[RawRow]) -> list[ChangelogEntry]:
    """Convert a list of raw rows into ChangelogEntry objects.

    Dispatches to the appropriate chunker based on the row's section.

    Args:
        rows: List of RawRow objects from the parser.

    Returns:
        List of successfully parsed ChangelogEntry objects.
    """
    entries: list[ChangelogEntry] = []
    decision_index = 0

    for row in rows:
        entry: ChangelogEntry | None = None

        if row.section == "changelog":
            entry = chunk_changelog_row(row)
        elif row.section == "key_decision":
            entry = chunk_decision_row(row, decision_index)
            decision_index += 1

        if entry is not None:
            entries.append(entry)

    logger.info(
        "Chunked rows into entries",
        extra={
            "total_rows": len(rows),
            "successful_entries": len(entries),
            "skipped": len(rows) - len(entries),
        },
    )
    return entries
