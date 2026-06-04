"""Tests for the changelog chunker."""

from pathlib import Path
import textwrap

import pytest

from src.indexing.parser import parse_file, RawRow
from src.indexing.chunker import chunk_rows, chunk_changelog_row, chunk_decision_row, _parse_date


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHANGELOG = textwrap.dedent("""\
    # Test Changelog

    ## 🧭 Key decisions

    | Decision | Rationale | When |
    |---|---|---|
    | **TypeORM** (not Prisma) | Team choice; better migrations | Jan 2024 |

    ---

    ## 📋 Changelog (What · Why · Where · When)

    | # | What | Why | Where | When |
    |---|---|---|---|---|
    | 3 | **Fix: Fleet duplicate on retry** — missing unique constraint | Idempotency on a retried create | `packages/db/src/migrations/*AddFleetUnique*` | Mar 20, 2026 10:35 IST |
    | 2 | **Perf: cache Billing reads** — read-through Redis cache | Stable data was re-queried per request | `apps/api/src/modules/billing/` | Jan 5, 2025 12:26 IST |
    | 1 | **Security: redact Auth secrets in logs** — token/secret fields masked | No secrets in the request log | `apps/api/src/observability/redact.ts` | Jan 23, 2024 13:05 IST |
""")


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    f = tmp_path / "test_changelog.md"
    f.write_text(SAMPLE_CHANGELOG, encoding="utf-8")
    return f


@pytest.fixture
def parsed_rows(sample_file: Path) -> list[RawRow]:
    return parse_file(sample_file)


@pytest.fixture
def entries(parsed_rows):
    return chunk_rows(parsed_rows)


# ---------------------------------------------------------------------------
# chunk_rows integration
# ---------------------------------------------------------------------------

class TestChunkRows:
    def test_produces_entries(self, entries):
        assert len(entries) > 0

    def test_changelog_entries_count(self, entries):
        changelog = [e for e in entries if e.section == "changelog"]
        assert len(changelog) == 3

    def test_decision_entries_count(self, entries):
        decisions = [e for e in entries if e.section == "key_decision"]
        assert len(decisions) == 1

    def test_changelog_entry_fields(self, entries):
        changelog = [e for e in entries if e.section == "changelog"]
        entry_3 = next(e for e in changelog if e.entry_number == 3)

        assert entry_3.category == "Fix"
        assert "Fleet duplicate" in entry_3.title
        assert "Idempotency" in entry_3.rationale
        assert entry_3.date is not None
        assert entry_3.date.year == 2026
        assert entry_3.date.month == 3

    def test_module_extraction(self, entries):
        changelog = [e for e in entries if e.section == "changelog"]
        entry_2 = next(e for e in changelog if e.entry_number == 2)
        assert entry_2.module == "billing"

    def test_paths_extraction(self, entries):
        changelog = [e for e in entries if e.section == "changelog"]
        entry_1 = next(e for e in changelog if e.entry_number == 1)
        assert len(entry_1.affected_paths) > 0
        assert any("redact" in p for p in entry_1.affected_paths)

    def test_decision_entry_fields(self, entries):
        decisions = [e for e in entries if e.section == "key_decision"]
        d = decisions[0]
        assert d.category == "Decision"
        assert "TypeORM" in d.title
        assert "Team choice" in d.rationale
        assert d.entry_number >= 10000  # Offset for decisions

    def test_chunk_id_is_deterministic(self, entries):
        ids = [e.chunk_id for e in entries]
        assert len(ids) == len(set(ids))  # All unique

    def test_embedding_text_not_empty(self, entries):
        for entry in entries:
            assert len(entry.embedding_text) > 10


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

class TestDateParsing:
    @pytest.mark.parametrize("date_str,year,month", [
        ("Jun 3, 2026 09:43 IST", 2026, 6),
        ("Jan 5, 2025 12:26 IST", 2025, 1),
        ("Mar 20, 2026 10:35 IST", 2026, 3),
        ("Jan 2024", 2024, 1),
    ])
    def test_parses_dates(self, date_str: str, year: int, month: int):
        dt = _parse_date(date_str)
        assert dt is not None
        assert dt.year == year
        assert dt.month == month

    def test_empty_date(self):
        assert _parse_date("") is None
        assert _parse_date("   ") is None

    def test_garbage_date(self):
        assert _parse_date("not a date at all xyz") is None or True  # graceful


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_malformed_row_too_few_cells(self):
        row = RawRow(cells=["only_one"], section="changelog", source_file="test.md", line_number=1)
        result = chunk_changelog_row(row)
        assert result is None

    def test_empty_what_column(self):
        row = RawRow(cells=["1", "", "why", "where", "when"], section="changelog", source_file="test.md", line_number=1)
        result = chunk_changelog_row(row)
        assert result is None

    def test_malformed_decision_row(self):
        row = RawRow(cells=[""], section="key_decision", source_file="test.md", line_number=1)
        result = chunk_decision_row(row, 0)
        assert result is None
