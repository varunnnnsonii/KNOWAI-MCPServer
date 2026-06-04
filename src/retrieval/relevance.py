"""Relevance validation for search results.

Implements a two-tier threshold system:
  1. Hard floor: drop results below a minimum similarity score (noise filter)
  2. Adaptive threshold: flag results scoring < 60% of top result as "low confidence"

This avoids returning garbage results while still surfacing marginally
relevant matches with appropriate warnings.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.observability import get_logger

logger = get_logger(__name__)


@dataclass
class ScoredResult:
    """A search result with relevance scoring and confidence metadata.

    Attributes:
        doc_id: The chunk/document ID.
        score: Combined relevance score (0.0–1.0, higher is better).
        is_low_confidence: Whether this result is below the adaptive threshold.
        metadata: Original metadata from the vector store.
        document: The embedding text.
    """

    doc_id: str
    score: float
    is_low_confidence: bool
    metadata: dict
    document: str


def validate_relevance(
    results: list[dict],
    hard_floor: float = 0.25,
    adaptive_ratio: float = 0.60,
) -> tuple[list[ScoredResult], str | None]:
    """Apply relevance thresholds to search results.

    Args:
        results: List of result dicts with keys: id, score, metadata, document.
        hard_floor: Minimum score to keep a result (0.0–1.0).
        adaptive_ratio: Results below this fraction of the top score
                        are flagged as low confidence.

    Returns:
        Tuple of:
        - List of ScoredResult objects (filtered and annotated).
        - Optional warning message if all results are low confidence.
    """
    if not results:
        return [], "No results found for your query. Try broader terms or different filters."

    # Apply hard floor
    filtered = [r for r in results if r["score"] >= hard_floor]

    if not filtered:
        logger.info(
            "All results below hard floor",
            extra={"threshold": hard_floor, "count": len(results)},
        )
        return [], (
            "Results found but confidence is too low. "
            "The changelog may not contain information about this topic."
        )

    # Determine adaptive threshold
    top_score = filtered[0]["score"]
    adaptive_threshold = top_score * adaptive_ratio

    scored_results: list[ScoredResult] = []
    low_confidence_count = 0

    for r in filtered:
        is_low = r["score"] < adaptive_threshold
        if is_low:
            low_confidence_count += 1

        scored_results.append(
            ScoredResult(
                doc_id=r["id"],
                score=r["score"],
                is_low_confidence=is_low,
                metadata=r.get("metadata", {}),
                document=r.get("document", ""),
            )
        )

    # Warning if all results are low confidence
    warning = None
    if low_confidence_count == len(scored_results):
        warning = (
            "Results found but confidence is low. "
            "The changelog may not contain information about this topic."
        )
    elif low_confidence_count > 0:
        warning = (
            f"{low_confidence_count} of {len(scored_results)} results have low confidence scores."
        )

    logger.info(
        "Relevance validation complete",
        extra={
            "total": len(results),
            "after_floor": len(filtered),
            "low_confidence": low_confidence_count,
            "top_score": round(top_score, 4),
            "threshold": round(adaptive_threshold, 4),
        },
    )

    return scored_results, warning
