"""Tests for the hybrid retrieval pipeline (Phase 4)."""

import math
from pathlib import Path

import pytest

from src.retrieval.bm25_store import BM25Store
from src.retrieval.relevance import validate_relevance
from src.retrieval.hybrid_search import HybridSearchEngine


# ---------------------------------------------------------------------------
# BM25Store tests
# ---------------------------------------------------------------------------

class TestBM25Store:
    def test_build_and_search(self):
        store = BM25Store()
        texts = [
            "fixed bug in billing module",
            "added new feature for authentication",
            "refactored the billing database schema",
        ]
        ids = ["id1", "id2", "id3"]
        store.build(doc_ids=ids, texts=texts)

        assert store.count == 3
        assert store.is_loaded

        # Search for "billing"
        results = store.search("billing")
        assert len(results) == 2
        # 'billing' appears in id1 and id3
        assert results[0][0] in ("id1", "id3")

    def test_save_and_load(self, tmp_path: Path):
        persist_path = tmp_path / "bm25.pkl"
        store1 = BM25Store(persist_path)
        store1.build(
            doc_ids=["id1", "id2", "id3"],
            texts=["test document for saving", "another random entry", "completely different text"],
        )
        store1.save()

        store2 = BM25Store(persist_path)
        assert store2.load() is True
        assert store2.count == 3
        results = store2.search("document")
        assert len(results) == 1
        assert results[0][0] == "id1"

    def test_search_empty(self):
        store = BM25Store()
        # No build called
        assert store.search("test") == []

    def test_empty_build(self):
        store = BM25Store()
        store.build([], [])
        assert store.count == 0
        assert store.search("test") == []


# ---------------------------------------------------------------------------
# Relevance Validation Tests
# ---------------------------------------------------------------------------

class TestRelevance:
    def test_hard_floor_filter(self):
        results = [
            {"id": "1", "score": 0.8, "metadata": {}, "document": "A"},
            {"id": "2", "score": 0.2, "metadata": {}, "document": "B"}, # Below 0.25 floor
        ]
        scored, warning = validate_relevance(results, hard_floor=0.25)
        assert len(scored) == 1
        assert scored[0].doc_id == "1"
        assert warning is None

    def test_all_below_floor(self):
        results = [
            {"id": "1", "score": 0.2, "metadata": {}, "document": "A"},
        ]
        scored, warning = validate_relevance(results, hard_floor=0.25)
        assert len(scored) == 0
        assert "confidence is too low" in warning

    def test_adaptive_threshold(self):
        results = [
            {"id": "1", "score": 0.9, "metadata": {}, "document": "A"}, # Top score
            {"id": "2", "score": 0.6, "metadata": {}, "document": "B"}, # > 60% of 0.9 (0.54)
            {"id": "3", "score": 0.4, "metadata": {}, "document": "C"}, # < 60% of 0.9 (0.54)
        ]
        scored, warning = validate_relevance(results, hard_floor=0.1, adaptive_ratio=0.6)
        
        assert len(scored) == 3
        assert scored[0].is_low_confidence is False
        assert scored[1].is_low_confidence is False
        assert scored[2].is_low_confidence is True
        
        assert warning is not None
        assert "1 of 3 results have low confidence" in warning


# ---------------------------------------------------------------------------
# Hybrid Search Mock Tests
# ---------------------------------------------------------------------------

class MockEmbedder:
    def __init__(self):
        self.model_name = "mock"
    def embed_query_as_list(self, q):
        return [0.1] * 384

class MockVectorStore:
    def __init__(self):
        self._count = 0
    def count(self): return self._count
    def query(self, query_embedding, n_results, where=None):
        if self._count == 0: return []
        # Return mock results
        return [
            {
                "id": "vec1", 
                "distance": 0.2, # score = 0.8
                "metadata": {"title": "Vec match", "module": "api"},
                "document": "vector document"
            }
        ]

class MockBM25Store:
    def __init__(self):
        self.is_loaded = False
    def search(self, query, n_results):
        if not self.is_loaded: return []
        return [("bm1", 2.5), ("vec1", 1.5)]


class TestHybridSearch:
    @pytest.fixture
    def engine(self):
        embedder = MockEmbedder()
        vstore = MockVectorStore()
        vstore._count = 10 # pretend we have data
        bstore = MockBM25Store()
        bstore.is_loaded = True
        return HybridSearchEngine(embedder, vstore, bstore, relevance_threshold=0.1)

    def test_empty_query(self, engine: HybridSearchEngine):
        res = engine.search("   ")
        assert "error" in res
        assert res["error_code"] == "EMPTY_QUERY"

    def test_short_query(self, engine: HybridSearchEngine):
        res = engine.search("") # less than 1 char (after strip)
        assert "error" in res
        assert res["error_code"] == "EMPTY_QUERY"

    def test_no_index(self):
        engine = HybridSearchEngine(MockEmbedder(), MockVectorStore(), MockBM25Store())
        res = engine.search("test query")
        assert "error" in res
        assert res["error_code"] == "NO_INDEX"

    def test_fusion_and_formatting(self, engine: HybridSearchEngine):
        res = engine.search("test query")
        
        assert "error" not in res
        results = res["results"]
        
        assert len(results) == 2
        # 'vec1' is in both vector and bm25, 'bm1' is only in bm25.
        # rrf score for vec1: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01612 = 0.0325
        # rrf score for bm1: 1/(60+1) = 0.01639
        # vec1 should rank first.
        
        assert results[0]["title"] == "Vec match"
        assert results[0]["module"] == "api"

    def test_metadata_filter(self, engine: HybridSearchEngine):
        # We pass a module filter
        res = engine.search("test", module="api")
        # In our mock, vec1 has module='api', bm1 has no metadata.
        # So bm1 should be filtered out by the post-filter.
        results = res["results"]
        assert len(results) == 1
        assert results[0]["module"] == "api"
