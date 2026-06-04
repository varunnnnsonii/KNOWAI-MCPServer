"""MCP Server configuration and entry point.

Registers tools and handles background indexing on startup.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Context

from src.config import get_settings
from src.indexing.indexer import Indexer
from src.observability import get_logger
from src.tools.search import register_tools

logger = get_logger(__name__)

# Single fastmcp instance
mcp = FastMCP(
    name="changelog-semantic-search",
    version="1.0.0",
    dependencies=["pydantic", "chromadb", "sentence-transformers", "rank-bm25"],
)

# Global indexer instance (injected into context/tools)
_indexer: Indexer | None = None


@mcp.server.lifespan
async def lifespan(server):
    """Lifecycle hook: runs on server startup and shutdown."""
    global _indexer
    logger.info("Starting MCP server lifecycle")

    try:
        # Load configuration
        settings = get_settings()

        # Initialize indexing pipeline
        _indexer = Indexer(settings)

        # Run background index
        # In a production environment, this could be offloaded to a separate task.
        # But for an MCP server, indexing the changelog at startup ensures the
        # data is ready before tools are called.
        logger.info("Running initial changelog indexing...")
        
        # We run it in a threadpool to not block the async event loop,
        # though during lifespan it doesn't strictly matter for FastMCP.
        await asyncio.to_thread(_indexer.run)
        
        logger.info("Indexing complete, ready for requests")

        # Yield control to the server
        yield

    except Exception as e:
        logger.error("Error during server lifecycle", extra={"error": str(e)})
        raise
    finally:
        logger.info("Shutting down MCP server")


# Register all tools from the tools module
register_tools(mcp)


def run_dev():
    """Run the server using mcp dev (requires mcp cli)."""
    # Simply delegates to mcp cli via fastmcp
    import subprocess
    subprocess.run(["mcp", "dev", "src.server:mcp"])


if __name__ == "__main__":
    # If run directly, run stdio (for MCP clients)
    mcp.run(transport="stdio")
