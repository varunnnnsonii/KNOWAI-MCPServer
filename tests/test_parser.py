"""Tests for the Markdown changelog parser."""

from pathlib import Path
import tempfile
import textwrap

import pytest

from src.indexing.parser import (
    parse_file,
    extract_category,
    extract_title,
    extract_module,
    extract_paths,
    discover_changelog_files,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHANGELOG = textwrap.dedent("""\
    # Test Changelog

    ## 🧭 Key decisions

    | Decision | Rationale | When |
    |---|---|---|
    | **TypeORM** (not Prisma) | Team choice; better migrations | Jan 2024 |
    | **REST everywhere** (no GraphQL) | Simpler one-protocol surface | Jan 2024 |

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
    """Create a temporary changelog file for testing."""
    f = tmp_path / "test_changelog.md"
    f.write_text(SAMPLE_CHANGELOG, encoding="utf-8")
    return f


@pytest.fixture
def sample_dir(tmp_path: Path, sample_file: Path) -> Path:
    """Return the directory containing the sample file."""
    return tmp_path


# ---------------------------------------------------------------------------
# discover_changelog_files
# ---------------------------------------------------------------------------

class TestDiscoverFiles:
    def test_finds_md_files(self, sample_dir: Path):
        files = discover_changelog_files(sample_dir)
        assert len(files) == 1
        assert files[0].suffix == ".md"

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            discover_changelog_files(tmp_path / "nonexistent")

    def test_empty_directory(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        files = discover_changelog_files(empty)
        assert files == []


# ---------------------------------------------------------------------------
# parse_file
# ---------------------------------------------------------------------------

class TestParseFile:
    def test_parses_changelog_rows(self, sample_file: Path):
        rows = parse_file(sample_file)
        changelog_rows = [r for r in rows if r.section == "changelog"]
        assert len(changelog_rows) == 3

    def test_parses_decision_rows(self, sample_file: Path):
        rows = parse_file(sample_file)
        decision_rows = [r for r in rows if r.section == "key_decision"]
        assert len(decision_rows) == 2

    def test_row_has_correct_cells(self, sample_file: Path):
        rows = parse_file(sample_file)
        changelog_rows = [r for r in rows if r.section == "changelog"]
        # First changelog row should have entry number 3
        first = changelog_rows[0]
        assert first.cells[0].strip() == "3"

    def test_row_tracks_source_file(self, sample_file: Path):
        rows = parse_file(sample_file)
        assert all(str(sample_file) in r.source_file for r in rows)

    def test_handles_nonexistent_file(self, tmp_path: Path):
        rows = parse_file(tmp_path / "missing.md")
        assert rows == []

    def test_handles_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")
        rows = parse_file(f)
        assert rows == []

    def test_handles_no_tables(self, tmp_path: Path):
        f = tmp_path / "no_tables.md"
        f.write_text("# Just a heading\n\nSome text.\n", encoding="utf-8")
        rows = parse_file(f)
        assert rows == []


# ---------------------------------------------------------------------------
# extract_category
# ---------------------------------------------------------------------------

class TestExtractCategory:
    @pytest.mark.parametrize("text,expected", [
        ("**Fix: Geofence duplicate on retry** — missing...", "Fix"),
        ("**Perf: cache Billing reads** — read-through...", "Perf"),
        ("**Security: redact Auth secrets** — token...", "Security"),
        ("**Refactor: narrow public API** — barrel...", "Refactor"),
        ("**Migration: extend vehicles** — added...", "Migration"),
        ("**Tests: edge cases** — empty/oversized...", "Tests"),
        ("**Worker: cleanup sweep** — repeatable...", "Worker"),
        ("**Deprecate: legacy endpoint** — old...", "Deprecate"),
        ("**Observability: metrics** — Prometheus...", "Observability"),
        ("**Driver schema** — entities...", "Feature"),
        ("**Shipment module** — service...", "Feature"),
    ])
    def test_extracts_category(self, text: str, expected: str):
        assert extract_category(text) == expected


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_bold_title(self):
        text = "**Fix: Geofence duplicate on retry** — missing unique constraint"
        assert extract_title(text) == "Fix: Geofence duplicate on retry"

    def test_no_bold(self):
        text = "Some change — with a description"
        assert extract_title(text) == "Some change"

    def test_bold_without_dash(self):
        text = "**Driver schema**"
        assert extract_title(text) == "Driver schema"


# ---------------------------------------------------------------------------
# extract_module
# ---------------------------------------------------------------------------

class TestExtractModule:
    @pytest.mark.parametrize("text,expected", [
        ("`apps/api/src/modules/fleet/fleet.service.ts`", "fleet"),
        ("`apps/api/src/modules/billing/`", "billing"),
        ("`apps/api/src/auth/auth.service.ts`", "auth"),
        ("`packages/db/src/migrations/*AddFleetUnique*`", "db"),
        ("`apps/api/src/modules/shipment/`", "shipment"),
        ("`apps/api/src/core/`", "core"),
        ("some text without paths", ""),
    ])
    def test_extracts_module(self, text: str, expected: str):
        assert extract_module(text) == expected


# ---------------------------------------------------------------------------
# extract_paths
# ---------------------------------------------------------------------------

class TestExtractPaths:
    def test_backtick_paths(self):
        text = "`apps/api/src/modules/fleet/`, `packages/db/`"
        paths = extract_paths(text)
        assert len(paths) == 2
        assert "apps/api/src/modules/fleet/" in paths

    def test_single_path(self):
        text = "`apps/api/src/auth/revocation.service.ts`"
        paths = extract_paths(text)
        assert len(paths) == 1

    def test_no_paths(self):
        text = ""
        paths = extract_paths(text)
        assert paths == []
