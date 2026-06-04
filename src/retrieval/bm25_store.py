"""BM25 keyword search index.

Provides keyword-based retrieval using the Okapi BM25 algorithm.
Used alongside vector search in the hybrid retrieval pipeline.

Supports:
  - Building index from ChangelogEntry embedding texts
  - Query-time scoring
  - Persistence via pickle serialization
  - Rebuild on demand
"""

from __future__ import annotations

import os
import pickle
import re
import tempfile
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.observability import get_logger

logger = get_logger(__name__)

# Simple tokenizer — lowercased word splitting
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens.

    Args:
        text: Input string.

    Returns:
        List of lowercase token strings.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """BM25 keyword search index backed by rank-bm25.

    Stores the tokenized corpus in memory with optional disk persistence.

    Attributes:
        persist_path: Path to the pickle persistence file.
    """

    def __init__(self, persist_path: Path | None = None):
        self.persist_path = persist_path
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        """Build the BM25 index from a list of documents.

        Args:
            doc_ids: Unique identifiers for each document.
            texts: The text content to tokenize and index.
        """
        if not texts:
            self._bm25 = None
            self._doc_ids = []
            self._corpus = []
            return

        start = time.monotonic()
        self._doc_ids = list(doc_ids)
        self._corpus = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus)
        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "BM25 index built",
            extra={
                "count": len(texts),
                "duration_ms": round(elapsed_ms, 1),
            },
        )

    def save(self) -> None:
        """Persist the BM25 index to disk via pickle.

        Uses atomic write (temp file + rename) to avoid corruption.
        """
        if self.persist_path is None:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "doc_ids": self._doc_ids,
            "corpus": self._corpus,
        }

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.persist_path.parent),
            prefix=".bm25_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(data, f)
            os.replace(tmp_path, str(self.persist_path))
            logger.info(
                "BM25 index saved",
                extra={"file": str(self.persist_path), "count": len(self._doc_ids)},
            )
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def load(self) -> bool:
        """Load a persisted BM25 index from disk.

        Returns:
            True if successfully loaded, False otherwise.
        """
        if self.persist_path is None or not self.persist_path.exists():
            return False

        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)

            self._doc_ids = data["doc_ids"]
            self._corpus = data["corpus"]

            if self._corpus:
                self._bm25 = BM25Okapi(self._corpus)
            else:
                self._bm25 = None

            logger.info(
                "BM25 index loaded from disk",
                extra={"count": len(self._doc_ids)},
            )
            return True

        except Exception as e:
            logger.warning(
                "Failed to load BM25 index",
                extra={"error": str(e), "file": str(self.persist_path)},
            )
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = 10) -> list[tuple[str, float]]:
        """Search the BM25 index.

        Args:
            query: The search query string.
            n_results: Maximum number of results to return.

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        if self._bm25 is None or not self._doc_ids:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        start = time.monotonic()
        scores = self._bm25.get_scores(tokens)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Pair scores with doc_ids and sort
        scored = [(self._doc_ids[i], float(scores[i])) for i in range(len(scores))]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter zero scores and limit
        results = [(doc_id, score) for doc_id, score in scored[:n_results] if score > 0]

        logger.info(
            "BM25 search complete",
            extra={
                "results_count": len(results),
                "duration_ms": round(elapsed_ms, 1),
                "top_score": round(results[0][1], 4) if results else 0.0,
            },
        )
        return results

    @property
    def count(self) -> int:
        """Number of documents in the index."""
        return len(self._doc_ids)

    @property
    def is_loaded(self) -> bool:
        """Whether the index has data."""
        return self._bm25 is not None and len(self._doc_ids) > 0
