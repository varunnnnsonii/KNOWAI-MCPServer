"""ChromaDB vector store wrapper.

Provides persistent vector storage with:
  - Idempotent upsert operations
  - Metadata filtering (module, category, date range)
  - Graceful error handling and corruption recovery
  - Collection lifecycle management
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from src.observability import get_logger

logger = get_logger(__name__)

# ChromaDB collection name
_COLLECTION_NAME = "changelog_entries"


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class VectorStore:
    """Persistent ChromaDB vector store for changelog embeddings.

    Attributes:
        persist_dir: Path to the ChromaDB persistence directory.
    """

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _init_client(self):
        """Initialize the ChromaDB client and collection."""
        import chromadb

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
            )
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Vector store initialized",
                extra={
                    "persist_dir": str(self.persist_dir),
                    "collection": _COLLECTION_NAME,
                    "count": self._collection.count(),
                },
            )
        except Exception as e:
            logger.error(
                "Failed to initialize vector store",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise VectorStoreError(
                f"Failed to initialize vector store: {e}"
            ) from e

    @property
    def collection(self):
        """Lazily initialize and return the ChromaDB collection."""
        if self._collection is None:
            self._init_client()
        return self._collection

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert entries into the vector store.

        Idempotent: re-upserting the same ID replaces the previous entry.

        Args:
            ids: Deterministic chunk IDs.
            embeddings: Embedding vectors.
            documents: Embedding text strings.
            metadatas: Flat metadata dicts (ChromaDB compatible).

        Raises:
            VectorStoreError: On ChromaDB failure.
        """
        if not ids:
            return

        try:
            start = time.monotonic()
            # ChromaDB has a batch limit — process in chunks of 5000
            batch_size = 5000
            for i in range(0, len(ids), batch_size):
                end = i + batch_size
                self.collection.upsert(
                    ids=ids[i:end],
                    embeddings=embeddings[i:end],
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                )
            elapsed_ms = (time.monotonic() - start) * 1000

            logger.info(
                "Vector store upsert complete",
                extra={
                    "entries_indexed": len(ids),
                    "duration_ms": round(elapsed_ms, 1),
                },
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to upsert entries: {e}"
            ) from e

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector store for similar entries.

        Args:
            query_embedding: The query vector.
            n_results: Maximum number of results.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of result dicts with keys: id, document, metadata, distance.

        Raises:
            VectorStoreError: On ChromaDB failure.
        """
        try:
            if self.count() == 0:
                return []

            start = time.monotonic()

            query_params: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(n_results, self.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_params["where"] = where

            results = self.collection.query(**query_params)
            elapsed_ms = (time.monotonic() - start) * 1000

            # Transform ChromaDB nested results into flat dicts
            output = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    output.append({
                        "id": doc_id,
                        "document": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 1.0,
                    })

            logger.info(
                "Vector store query complete",
                extra={
                    "results_count": len(output),
                    "duration_ms": round(elapsed_ms, 1),
                    "top_score": round(1.0 - output[0]["distance"], 4) if output else 0.0,
                },
            )
            return output

        except Exception as e:
            raise VectorStoreError(
                f"Vector store query failed: {e}"
            ) from e

    def delete_by_source(self, source_file: str) -> None:
        """Delete all entries from a specific source file.

        Used during incremental re-indexing when a file changes or is deleted.

        Args:
            source_file: The source_file metadata value to match.
        """
        try:
            # Get all IDs with this source file
            existing = self.collection.get(
                where={"source_file": source_file},
                include=[],
            )
            if existing and existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                logger.info(
                    "Deleted entries for source file",
                    extra={
                        "source_file": source_file,
                        "count": len(existing["ids"]),
                    },
                )
        except Exception as e:
            logger.warning(
                "Failed to delete entries by source",
                extra={"source_file": source_file, "error": str(e)},
            )

    def count(self) -> int:
        """Return the total number of entries in the store."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Delete the entire collection and recreate it.

        Used for corruption recovery or full re-index.
        """
        try:
            if self._client is not None:
                self._client.delete_collection(_COLLECTION_NAME)
                self._collection = self._client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            logger.info("Vector store reset")
        except Exception as e:
            # Nuclear option: delete the directory and re-init
            logger.warning(
                "Collection reset failed, removing persistence directory",
                extra={"error": str(e)},
            )
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)
            self._client = None
            self._collection = None
            self._init_client()

    def get_all_ids(self) -> list[str]:
        """Return all document IDs currently in the store."""
        try:
            result = self.collection.get(include=[])
            return result["ids"] if result and result["ids"] else []
        except Exception:
            return []
