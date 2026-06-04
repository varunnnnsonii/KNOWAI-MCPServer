"""Shared mutable state for the MCP server.

This module holds the global Indexer reference so that tools can access it
without importing src.server (which breaks under `mcp dev` due to module
identity issues with importlib).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.indexing.indexer import Indexer

# Set by src.server.lifespan() after indexing completes.
indexer: Indexer | None = None
