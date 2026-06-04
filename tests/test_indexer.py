"""Tests for the indexing pipeline — embedder, vector store, file tracker, indexer."""

from pathlib import Path
import json
import textwrap
import time

import pytest

from src.indexing.file_tracker import FileManifest, compute_file_hash, detect_changes
from src.indexing.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.indexing.indexer import Indexer
from src.config import Settings


# ---------------------------------------------------------------------------
# Sample changelog for integration tests
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
def sample_changelog_dir(tmp_path: Path) -> Path:
    """Create a temporary changelog directory with a sample file."""
    changelog_dir = tmp_path / "changelogs"
    changelog_dir.mkdir()
    (changelog_dir / "test_changelog.md").write_text(SAMPLE_CHANGELOG, encoding="utf-8")
    return changelog_dir


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def settings(tmp_path: Path, sample_changelog_dir: Path, data_dir: Path) -> Settings:
    """Create Settings pointing to temp directories."""
    return Settings(
        changelog_dir=sample_changelog_dir,
        data_dir=data_dir,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


# ---------------------------------------------------------------------------
# FileManifest tests
# ---------------------------------------------------------------------------

class TestFileManifest:
    def test_empty_manifest(self, tmp_path: Path):
        m = FileManifest(tmp_path / "manifest.json")
        assert m.get_all_tracked_files() == set()
        assert m.total_entries == 0

    def test_update_and_retrieve(self, tmp_path: Path):
        m = FileManifest(tmp_path / "manifest.json")
        m.update_entry("/path/to/file.md", "abc123", 42)
        entry = m.get_entry("/path/to/file.md")
        assert entry is not None
        assert entry["file_hash"] == "abc123"
        assert entry["entry_count"] == 42
        assert "last_indexed" in entry

    def test_save_and_reload(self, tmp_path: Path):
        path = tmp_path / "manifest.json"
        m1 = FileManifest(path)
        m1.update_entry("/a.md", "hash1", 10)
        m1.update_entry("/b.md", "hash2", 20)
        m1.save()

        # Reload from disk
        m2 = FileManifest(path)
        assert len(m2.get_all_tracked_files()) == 2
        assert m2.total_entries == 30

    def test_remove_entry(self, tmp_path: Path):
        m = FileManifest(tmp_path / "manifest.json")
        m.update_entry("/a.md", "hash1", 10)
        m.remove_entry("/a.md")
        assert m.get_entry("/a.md") is None

    def test_corrupt_manifest_recovers(self, tmp_path: Path):
        path = tmp_path / "manifest.json"
        path.write_text("NOT VALID JSON!!!", encoding="utf-8")
        m = FileManifest(path)
        assert m.get_all_tracked_files() == set()  # Started fresh


# ---------------------------------------------------------------------------
# compute_file_hash tests
# ---------------------------------------------------------------------------

class TestComputeFileHash:
    def test_deterministic_hash(self, tmp_path: Path):
        f = tmp_path / "test.md"
        f.write_text("hello world", encoding="utf-8")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("content A", encoding="utf-8")
        f2.write_text("content B", encoding="utf-8")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            compute_file_hash(tmp_path / "nonexistent.md")


# ---------------------------------------------------------------------------
# detect_changes tests
# ---------------------------------------------------------------------------

class TestDetectChanges:
    def test_all_new_files(self, tmp_path: Path):
        manifest = FileManifest(tmp_path / "manifest.json")
        f = tmp_path / "new.md"
        f.write_text("content", encoding="utf-8")
        new, unchanged, deleted = detect_changes(manifest, [f])
        assert len(new) == 1
        assert len(unchanged) == 0
        assert len(deleted) == 0

    def test_unchanged_file(self, tmp_path: Path):
        f = tmp_path / "test.md"
        f.write_text("content", encoding="utf-8")

        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.update_entry(str(f), compute_file_hash(f), 5)

        new, unchanged, deleted = detect_changes(manifest, [f])
        assert len(new) == 0
        assert len(unchanged) == 1

    def test_changed_file(self, tmp_path: Path):
        f = tmp_path / "test.md"
        f.write_text("original", encoding="utf-8")

        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.update_entry(str(f), compute_file_hash(f), 5)

        # Modify file
        f.write_text("modified content", encoding="utf-8")

        new, unchanged, deleted = detect_changes(manifest, [f])
        assert len(new) == 1
        assert len(unchanged) == 0

    def test_deleted_file(self, tmp_path: Path):
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.update_entry("/old/file.md", "oldhash", 10)

        new, unchanged, deleted = detect_changes(manifest, [])
        assert len(deleted) == 1
        assert "/old/file.md" in deleted


# ---------------------------------------------------------------------------
# Embedder tests (uses real model — marked slow)
# ---------------------------------------------------------------------------

class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        """Shared embedder instance (model is expensive to load)."""
        return Embedder("sentence-transformers/all-MiniLM-L6-v2")

    @pytest.mark.slow
    def test_embed_texts_returns_vectors(self, embedder: Embedder):
        vectors = embedder.embed_texts(["hello world", "test query"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 384  # MiniLM dimension

    @pytest.mark.slow
    def test_embed_query_cached(self, embedder: Embedder):
        v1 = embedder.embed_query("test query")
        v2 = embedder.embed_query("test query")
        assert v1 == v2  # Same object from cache

    @pytest.mark.slow
    def test_embed_empty_list(self, embedder: Embedder):
        vectors = embedder.embed_texts([])
        assert vectors == []

    @pytest.mark.slow
    def test_dimension_property(self, embedder: Embedder):
        assert embedder.dimension == 384


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> VectorStore:
        return VectorStore(tmp_path / "chroma_test")

    def test_upsert_and_count(self, store: VectorStore):
        store.upsert(
            ids=["id1", "id2"],
            embeddings=[[0.1] * 384, [0.2] * 384],
            documents=["doc1", "doc2"],
            metadatas=[{"source_file": "a.md"}, {"source_file": "a.md"}],
        )
        assert store.count() == 2

    def test_upsert_idempotent(self, store: VectorStore):
        store.upsert(
            ids=["id1"],
            embeddings=[[0.1] * 384],
            documents=["doc1"],
            metadatas=[{"source_file": "a.md"}],
        )
        store.upsert(
            ids=["id1"],
            embeddings=[[0.2] * 384],
            documents=["doc1 updated"],
            metadatas=[{"source_file": "a.md"}],
        )
        assert store.count() == 1  # Still 1, not 2

    def test_query_returns_results(self, store: VectorStore):
        store.upsert(
            ids=["id1"],
            embeddings=[[0.1] * 384],
            documents=["billing cache fix"],
            metadatas=[{"source_file": "a.md", "module": "billing"}],
        )
        results = store.query(query_embedding=[0.1] * 384, n_results=5)
        assert len(results) == 1
        assert results[0]["id"] == "id1"

    def test_query_empty_store(self, store: VectorStore):
        results = store.query(query_embedding=[0.1] * 384, n_results=5)
        assert results == []

    def test_delete_by_source(self, store: VectorStore):
        store.upsert(
            ids=["id1", "id2"],
            embeddings=[[0.1] * 384, [0.2] * 384],
            documents=["doc1", "doc2"],
            metadatas=[{"source_file": "a.md"}, {"source_file": "b.md"}],
        )
        store.delete_by_source("a.md")
        assert store.count() == 1

    def test_reset(self, store: VectorStore):
        store.upsert(
            ids=["id1"],
            embeddings=[[0.1] * 384],
            documents=["doc1"],
            metadatas=[{"source_file": "a.md"}],
        )
        store.reset()
        assert store.count() == 0

    def test_get_all_ids(self, store: VectorStore):
        store.upsert(
            ids=["id1", "id2"],
            embeddings=[[0.1] * 384, [0.2] * 384],
            documents=["doc1", "doc2"],
            metadatas=[{"source_file": "a.md"}, {"source_file": "a.md"}],
        )
        ids = store.get_all_ids()
        assert set(ids) == {"id1", "id2"}

    def test_upsert_empty(self, store: VectorStore):
        store.upsert(ids=[], embeddings=[], documents=[], metadatas=[])
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Indexer integration tests (uses real model — marked slow)
# ---------------------------------------------------------------------------

class TestIndexer:
    @pytest.mark.slow
    def test_full_index_pipeline(self, settings: Settings):
        """T13: Full index → search pipeline returns results."""
        indexer = Indexer(settings)
        entries = indexer.run()
        assert len(entries) > 0
        assert indexer.vector_store.count() > 0

    @pytest.mark.slow
    def test_incremental_skip_unchanged(self, settings: Settings):
        """T15: Incremental re-index skips unchanged files."""
        indexer = Indexer(settings)
        entries1 = indexer.run()
        count1 = len(entries1)

        # Run again — should skip all files
        entries2 = indexer.run()
        # No new entries processed (all unchanged)
        assert len(entries2) == 0
        # But store still has all entries
        assert indexer.vector_store.count() == count1

    @pytest.mark.slow
    def test_incremental_detect_changes(self, settings: Settings, sample_changelog_dir: Path):
        """T14: Incremental re-index detects changes."""
        indexer = Indexer(settings)
        indexer.run()
        count1 = indexer.vector_store.count()

        # Modify the file
        changelog_file = sample_changelog_dir / "test_changelog.md"
        content = changelog_file.read_text(encoding="utf-8")
        content += "\n| 4 | **Fix: New bug fix** — something | Reason | `apps/api/` | Jun 1, 2026 |\n"
        changelog_file.write_text(content, encoding="utf-8")

        # Re-index — should detect change
        entries = indexer.run()
        assert len(entries) > 0

    @pytest.mark.slow
    def test_deleted_file_cleanup(self, settings: Settings, sample_changelog_dir: Path):
        """T16: Deleted file entries removed from index."""
        indexer = Indexer(settings)
        indexer.run()
        assert indexer.vector_store.count() > 0

        # Delete the file
        (sample_changelog_dir / "test_changelog.md").unlink()

        # Re-index — should clean up
        indexer.run()
        assert indexer.vector_store.count() == 0

    @pytest.mark.slow
    def test_idempotent_indexing(self, settings: Settings):
        """T20: Re-indexing is idempotent (same results after double-index)."""
        indexer = Indexer(settings)
        indexer.run()
        count1 = indexer.vector_store.count()

        # Force re-index by clearing manifest
        indexer.manifest = FileManifest(settings.manifest_path.parent / "fresh_manifest.json")
        indexer.run()
        count2 = indexer.vector_store.count()

        assert count1 == count2

    @pytest.mark.slow
    def test_index_status(self, settings: Settings):
        indexer = Indexer(settings)
        indexer.run()
        status = indexer.get_status()
        assert status["total_entries"] > 0
        assert status["tracked_files"] > 0
        assert "model_name" in status

    @pytest.mark.slow
    def test_empty_directory(self, tmp_path: Path):
        """No crash when changelog directory is empty."""
        empty_dir = tmp_path / "empty_changelogs"
        empty_dir.mkdir()
        s = Settings(
            changelog_dir=empty_dir,
            data_dir=tmp_path / "data",
        )
        indexer = Indexer(s)
        entries = indexer.run()
        assert entries == []

    @pytest.mark.slow
    def test_persistence_across_restarts(self, settings: Settings):
        """T19: Index persists across restarts."""
        indexer1 = Indexer(settings)
        indexer1.run()
        count1 = indexer1.vector_store.count()
        assert count1 > 0

        # Simulate restart: create a new Indexer with same settings
        indexer2 = Indexer(settings)
        count2 = indexer2.vector_store.count()
        assert count2 == count1
