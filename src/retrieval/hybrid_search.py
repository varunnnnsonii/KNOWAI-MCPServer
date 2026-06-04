"""Hybrid search — Vector + BM25 + Reciprocal Rank Fusion.

Retrieval pipeline:
  Query → preprocessing → parallel retrieval (vector + BM25)
       → RRF fusion → relevance filtering → ranked results

RRF is score-agnostic: it fuses ranked lists by reciprocal rank position,
avoiding the need to normalize cosine similarity vs BM25 scores.
"""

from __future__ import annotations

import time
from typing import Any

from src.indexing.embedder import Embedder, EmbeddingError
from src.observability import get_logger
from src.retrieval.bm25_store import BM25Store
from src.retrieval.relevance import ScoredResult, validate_relevance
from src.retrieval.vector_store import VectorStore, VectorStoreError

logger = get_logger(__name__)

# RRF constant (standard value from the original Cormack et al. paper)
_RRF_K = 60

# Maximum query length before truncation
_MAX_QUERY_LENGTH = 500

# Minimum query length for meaningful search
_MIN_QUERY_LENGTH = 1


class SearchError(Exception):
    """Raised when search operations fail."""


class HybridSearchEngine:
    """Hybrid retrieval engine combining vector search and BM25.

    Attributes:
        embedder: Embedding model for query vectorization.
        vector_store: ChromaDB vector store.
        bm25_store: BM25 keyword index.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        relevance_threshold: float = 0.25,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.relevance_threshold = relevance_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 5,
        module: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Execute a hybrid search over the changelog index.

        Args:
            query: Natural language search query.
            limit: Maximum number of results to return.
            module: Optional module name filter.
            category: Optional category filter.

        Returns:
            Dict with keys: results, warning, query_info.
        """
        start = time.monotonic()

        # --- Input validation ---
        query = query.strip()
        if not query:
            return self._error_response(
                "Please provide a search query.",
                "EMPTY_QUERY",
                "Provide a natural language query describing what you're looking for.",
            )

        if len(query) < _MIN_QUERY_LENGTH:
            return self._error_response(
                "Query too short. Please provide a more descriptive search query.",
                "SHORT_QUERY",
                "Try a query like 'What security changes were made?' or 'Why was TypeORM chosen?'",
            )

        # Truncate long queries
        original_length = len(query)
        if len(query) > _MAX_QUERY_LENGTH:
            query = query[:_MAX_QUERY_LENGTH]
            logger.info(
                "Query truncated",
                extra={"original_length": original_length, "truncated_to": _MAX_QUERY_LENGTH},
            )

        # Check if index has data
        if self.vector_store.count() == 0 and not self.bm25_store.is_loaded:
            return self._error_response(
                "No changelog data has been indexed yet. Please add changelog files and re-index.",
                "NO_INDEX",
                "Add markdown changelog files to the data/changelogs/ directory.",
            )

        # --- Build metadata filter ---
        where_filter = self._build_filter(module, category)

        # --- Parallel retrieval ---
        vector_results = self._vector_search(query, limit * 4, where_filter)
        bm25_results = self._bm25_search(query, limit * 4)

        # --- RRF Fusion ---
        fused = self._rrf_fusion(vector_results, bm25_results)

        # --- Apply metadata filters to BM25 results (BM25 doesn't support metadata) ---
        if where_filter:
            fused = self._apply_metadata_filter(fused, module, category)

        # --- Relevance filtering ---
        scored_results, warning = validate_relevance(
            fused[:limit * 2],
            hard_floor=self.relevance_threshold,
        )

        # Limit final results
        scored_results = scored_results[:limit]

        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Hybrid search complete",
            extra={
                "query_length": len(query),
                "results_count": len(scored_results),
                "duration_ms": round(elapsed_ms, 1),
                "top_score": round(scored_results[0].score, 4) if scored_results else 0.0,
                "filters": {"module": module, "category": category},
            },
        )

        return {
            "results": [self._format_result(r) for r in scored_results],
            "warning": warning,
            "query_info": {
                "original_length": original_length,
                "truncated": original_length > _MAX_QUERY_LENGTH,
                "vector_candidates": len(vector_results),
                "bm25_candidates": len(bm25_results),
                "fused_candidates": len(fused),
                "duration_ms": round(elapsed_ms, 1),
            },
        }

    # ------------------------------------------------------------------
    # Retrieval methods
    # ------------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        n_results: int,
        where_filter: dict | None,
    ) -> list[dict[str, Any]]:
        """Run vector similarity search."""
        try:
            query_embedding = self.embedder.embed_query_as_list(query)
            raw_results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=n_results,
                where=where_filter,
            )
            # Convert distance to similarity score (cosine distance → similarity)
            return [
                {
                    "id": r["id"],
                    "score": max(0.0, 1.0 - r["distance"]),
                    "metadata": r["metadata"],
                    "document": r["document"],
                    "source": "vector",
                }
                for r in raw_results
            ]
        except (EmbeddingError, VectorStoreError) as e:
            logger.warning(
                "Vector search failed, falling back to BM25-only",
                extra={"error": str(e)},
            )
            return []

    def _bm25_search(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """Run BM25 keyword search."""
        if not self.bm25_store.is_loaded:
            return []

        try:
            raw_results = self.bm25_store.search(query, n_results)
            if not raw_results:
                return []

            # Normalize BM25 scores to 0-1 range
            max_score = raw_results[0][1] if raw_results else 1.0
            if max_score == 0:
                max_score = 1.0

            return [
                {
                    "id": doc_id,
                    "score": score / max_score,  # Normalize to [0, 1]
                    "metadata": {},  # BM25 doesn't carry metadata; will be filled by fusion
                    "document": "",
                    "source": "bm25",
                }
                for doc_id, score in raw_results
            ]
        except Exception as e:
            logger.warning(
                "BM25 search failed",
                extra={"error": str(e)},
            )
            return []

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict[str, Any]]:
        """Merge results using Reciprocal Rank Fusion.

        RRF score for document d = Σ 1/(k + rank_i(d))
        where k=60 and rank_i is the rank in each result list.

        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.

        Returns:
            Fused results sorted by combined RRF score.
        """
        rrf_scores: dict[str, float] = {}
        metadata_map: dict[str, dict] = {}
        document_map: dict[str, str] = {}

        # Score from vector results
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)
            metadata_map[doc_id] = result["metadata"]
            document_map[doc_id] = result["document"]

        # Score from BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)
            # Only overwrite metadata if we don't have it from vector
            if doc_id not in metadata_map:
                metadata_map[doc_id] = result["metadata"]
                document_map[doc_id] = result["document"]

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

        # Normalize RRF scores: scale by max_rrf so the top result is always 1.0.
        # This preserves relative differences without pushing the lowest result to 0.0.
        max_rrf = rrf_scores[sorted_ids[0]] if sorted_ids else 1.0
        if max_rrf == 0.0:
            max_rrf = 1.0

        return [
            {
                "id": doc_id,
                "score": rrf_scores[doc_id] / max_rrf,
                "metadata": metadata_map.get(doc_id, {}),
                "document": document_map.get(doc_id, ""),
            }
            for doc_id in sorted_ids
        ]

    # ------------------------------------------------------------------
    # Metadata filtering (for BM25 results that lack native filtering)
    # ------------------------------------------------------------------

    def _apply_metadata_filter(
        self,
        results: list[dict],
        module: str | None,
        category: str | None,
    ) -> list[dict]:
        """Post-filter results by metadata (for BM25 results without native filtering)."""
        filtered = []
        for r in results:
            meta = r.get("metadata", {})
            if module and meta.get("module", "").lower() != module.lower():
                continue
            if category and meta.get("category", "").lower() != category.lower():
                continue
            filtered.append(r)
        return filtered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        module: str | None,
        category: str | None,
    ) -> dict | None:
        """Build a ChromaDB where-filter from optional parameters."""
        conditions = []

        if module:
            conditions.append({"module": {"$eq": module}})
        if category:
            conditions.append({"category": {"$eq": category}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _format_result(self, scored: ScoredResult) -> dict[str, Any]:
        """Format a ScoredResult into the output schema."""
        meta = scored.metadata
        paths_raw = meta.get("affected_paths", "")
        affected_paths = paths_raw.split("|") if paths_raw else []

        return {
            "entry_number": meta.get("entry_number", 0),
            "title": meta.get("title", ""),
            "description": scored.document,
            "rationale": meta.get("rationale", ""),
            "category": meta.get("category", ""),
            "module": meta.get("module", ""),
            "affected_paths": affected_paths,
            "date": meta.get("date_str", ""),
            "relevance_score": round(scored.score, 4),
            "source_file": meta.get("source_file", ""),
            "section": meta.get("section", ""),
            "is_low_confidence": scored.is_low_confidence,
        }

    def _error_response(
        self,
        message: str,
        error_code: str,
        suggestion: str,
    ) -> dict[str, Any]:
        """Build a standardized error response."""
        return {
            "results": [],
            "error": message,
            "error_code": error_code,
            "suggestion": suggestion,
            "warning": None,
        }
