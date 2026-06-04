"""MCP tools for changelog search.

Registers tool schemas and executes searches via the hybrid search engine.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from src.observability import get_logger
from src.retrieval.hybrid_search import HybridSearchEngine

logger = get_logger(__name__)


def get_search_engine() -> HybridSearchEngine:
    """Helper to lazily construct the search engine using the global Indexer.
    
    In FastMCP, tools run synchronously or asynchronously, but they need
    access to the state initialized in `src/server.py`.
    """
    import src._state as state
    indexer = state.indexer
    
    if indexer is None:
        raise RuntimeError("Indexer not initialized. Server lifecycle failed.")

    # We build the BM25 store from the vector store on demand if needed,
    # or rely on the fact that BM25 could be persisted. But wait, we didn't
    # populate BM25 in Indexer.run(). Let's fix that.
    # The hybrid search engine needs BM25!
    
    # Actually, we should initialize the BM25 store during indexing.
    # Let's ensure it's loaded.
    from src.retrieval.bm25_store import BM25Store
    bm25 = BM25Store(indexer.settings.data_dir / "bm25.pkl")
    
    if not bm25.is_loaded:
        bm25.load()
        if not bm25.is_loaded:
            # Rebuild it if it's missing (e.g. first run)
            ids = indexer.vector_store.get_all_ids()
            if ids:
                # We need texts to rebuild BM25... but they aren't stored fully,
                # wait, they ARE stored in vector store `document` field!
                res = indexer.vector_store.collection.get(ids=ids, include=["documents"])
                if res and res["documents"]:
                    # ChromaDB may return None for some documents; filter them out
                    clean_ids = []
                    clean_texts = []
                    for doc_id, text in zip(res["ids"], res["documents"]):
                        if text is not None:
                            clean_ids.append(doc_id)
                            clean_texts.append(text)
                    if clean_ids:
                        bm25.build(clean_ids, clean_texts)
                        bm25.save()

    return HybridSearchEngine(
        embedder=indexer.embedder,
        vector_store=indexer.vector_store,
        bm25_store=bm25,
    )


def register_tools(mcp: FastMCP) -> None:
    """Register all search tools with the FastMCP app."""

    @mcp.tool(
        name="search_changelog",
        description="Search the developer changelog for specific changes, features, bug fixes, or updates."
    )
    def search_changelog(
        query: str = Field(
            description="Natural language query describing what you're looking for (e.g., 'What billing changes were made in January?')"
        ),
        module: Optional[str] = Field(
            default=None,
            description="Optional filter by exact module name (e.g., 'api', 'db', 'frontend')."
        ),
        version: Optional[int] = Field(
            default=None,
            description="Optional filter by specific version or entry number (e.g., 600)."
        ),
        start_date: Optional[str] = Field(
            default=None,
            description="Optional start date filter (ISO format or similar)."
        ),
        end_date: Optional[str] = Field(
            default=None,
            description="Optional end date filter (ISO format or similar)."
        ),
        limit: int = Field(
            default=5,
            description="Maximum number of results to return. Default 5, max 10."
        ),
    ) -> str:
        """Search the changelog."""
        limit = min(limit, 10)
        logger.info("Executing search_changelog tool", extra={"query": query, "module_name": module})
        
        try:
            engine = get_search_engine()
            result = engine.search(
                query, 
                limit=limit, 
                module=module, 
                version=version, 
                start_date=start_date, 
                end_date=end_date
            )
            return _format_search_response(result)
        except Exception as e:
            logger.error("search_changelog tool failed", extra={"error": str(e)})
            return _friendly_error_message(e)

    @mcp.tool(
        name="search_decision_rationale",
        description="Search architectural and technical decisions to understand *why* a specific approach was taken."
    )
    def search_decision_rationale(
        query: str = Field(
            description="Natural language query describing the architectural component or decision (e.g., 'Why did we choose TypeORM over Prisma?')"
        ),
        limit: int = Field(
            default=3,
            description="Maximum number of results to return. Default 3, max 10."
        ),
    ) -> str:
        """Search the key decisions."""
        limit = min(limit, 10)
        logger.info("Executing search_decision_rationale tool", extra={"query": query})
        
        try:
            engine = get_search_engine()
            # Filter specifically to "Decision" category (assigned by chunker)
            result = engine.search(query, limit=limit, category="Decision")
            return _format_search_response(result)
        except Exception as e:
            logger.error("search_decision_rationale tool failed", extra={"error": str(e)})
            return _friendly_error_message(e)

    @mcp.tool(
        name="get_index_status",
        description="Get the current status of the changelog index, including number of entries and tracked files."
    )
    def get_index_status() -> str:
        """Get index status."""
        logger.info("Executing get_index_status tool")
        try:
            import src._state as state
            indexer = state.indexer
            if indexer is None:
                return "Index is not initialized."
            
            status = indexer.get_status()
            
            out = [
                "--- Index Status ---",
                f"Total Entries: {status.get('total_entries')}",
                f"Tracked Files: {status.get('tracked_files')}",
                f"Model Name: {status.get('model_name')}",
                "",
                "Metadata:",
            ]
            for k, v in status.get("index_metadata", {}).items():
                out.append(f"  {k}: {v}")
                
            return "\n".join(out)
        except Exception as e:
            logger.error("get_index_status tool failed", extra={"error": str(e)})
            return f"Error getting status: {str(e)}"


def _format_search_response(search_result: dict) -> str:
    """Format the dictionary output from HybridSearchEngine into a readable string."""
    if "error" in search_result:
        return f"Error ({search_result.get('error_code')}): {search_result['error']}\nSuggestion: {search_result.get('suggestion')}"

    results = search_result.get("results", [])
    if not results:
        warning = search_result.get("warning")
        return warning or "No relevant changelog entries found."

    output = []
    
    warning = search_result.get("warning")
    if warning:
        output.append(f"WARNING: {warning}\n")

    for i, r in enumerate(results, 1):
        output.append(f"--- Result {i} (Score: {r.get('relevance_score')}) ---")
        output.append(f"Version/Entry: {r.get('entry_number')}")
        output.append(f"Title: {r.get('title')}")
        output.append(f"Category: {r.get('category')} | Module: {r.get('module')}")
        output.append(f"Date: {r.get('date')}")
        
        if r.get("rationale"):
            output.append(f"Rationale: {r.get('rationale')}")
        
        output.append(f"Description:\n{r.get('description')}")
        
        paths = r.get("affected_paths", [])
        if paths:
            output.append(f"Affected paths: {', '.join(paths)}")
            
        output.append("") # blank line

    return "\n".join(output)

def _friendly_error_message(e: Exception) -> str:
    """Map technical exceptions to user-friendly explanations."""
    err_str = str(e).lower()
    if "connection refused" in err_str or "connect" in err_str:
        return "The search service is currently unavailable. Please check your connection to the vector database or embedding provider."
    elif "index" in err_str or "collection" in err_str:
        return "The changelog index could not be accessed. It may still be building or requires a re-index."
    elif "embedding" in err_str:
        return "Failed to process the search query. The embedding model might be offline or encountering an issue."
    else:
        return "An unexpected error occurred while processing your search. Please try rephrasing your query or checking the server logs for more details."
