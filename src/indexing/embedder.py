"""Embedding model abstraction.

Provides a thin wrapper around sentence-transformers for:
  - Lazy model loading (defer heavy download until first use)
  - Batch embedding at index time
  - LRU cache for query-time embeddings
  - Graceful failure with clear error messages

The model is configurable via Settings.embedding_model.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING

import numpy as np

from src.observability import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

# Maximum retries for model download/load
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2.0


class EmbeddingError(Exception):
    """Raised when embedding operations fail."""


class Embedder:
    """Sentence-transformer embedding model with lazy loading.

    Attributes:
        model_name: HuggingFace model identifier.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        """Load the model with retry logic.

        Returns:
            Loaded SentenceTransformer instance.

        Raises:
            EmbeddingError: If the model fails to load after retries.
        """
        from sentence_transformers import SentenceTransformer

        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(
                    "Loading embedding model",
                    extra={"model_name": self.model_name, "retry_count": attempt},
                )
                start = time.monotonic()
                model = SentenceTransformer(self.model_name)
                elapsed_ms = (time.monotonic() - start) * 1000

                # Determine dimensionality from a test embedding
                test_vec = model.encode(["test"], show_progress_bar=False)
                self._dimension = test_vec.shape[1]

                logger.info(
                    "Embedding model loaded",
                    extra={
                        "model_name": self.model_name,
                        "dimension": self._dimension,
                        "duration_ms": round(elapsed_ms, 1),
                    },
                )
                return model

            except Exception as e:
                last_error = e
                logger.warning(
                    "Model load attempt failed",
                    extra={
                        "model_name": self.model_name,
                        "retry_count": attempt,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_SECONDS * attempt)

        raise EmbeddingError(
            f"Failed to load embedding model '{self.model_name}' after {_MAX_RETRIES} attempts. "
            f"Check your internet connection for first-time setup. Error: {last_error}"
        )

    @property
    def model(self) -> SentenceTransformer:
        """Lazily load and return the embedding model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (loads model if needed)."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        assert self._dimension is not None
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Embedding operations
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            return []

        try:
            start = time.monotonic()
            vectors = self.model.encode(
                texts,
                show_progress_bar=False,
                batch_size=64,
                normalize_embeddings=True,
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            logger.info(
                "Batch embedding complete",
                extra={
                    "count": len(texts),
                    "duration_ms": round(elapsed_ms, 1),
                },
            )
            return vectors.tolist()

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings: {e}"
            ) from e

    @functools.lru_cache(maxsize=128)
    def embed_query(self, query: str) -> tuple[float, ...]:
        """Embed a single query with LRU caching.

        Uses a tuple return type for hashability (LRU cache requirement).

        Args:
            query: The search query string.

        Returns:
            Embedding vector as a tuple of floats.

        Raises:
            EmbeddingError: If embedding fails.
        """
        try:
            vector = self.model.encode(
                [query],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return tuple(vector[0].tolist())

        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed query: {e}"
            ) from e

    def embed_query_as_list(self, query: str) -> list[float]:
        """Embed a single query, returning a list (for ChromaDB compatibility).

        Args:
            query: The search query string.

        Returns:
            Embedding vector as a list of floats.
        """
        return list(self.embed_query(query))
