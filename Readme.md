# KNOWAI Semantic Search MCP Server

this is a public repo linked/should be linked to https://github.com/varunnnnsonii/KNOWAI-MCPServer

## Overview

The **Semantic Search MCP Server** is a Python-based Model Context Protocol (MCP) server designed to perform semantic search over changelog history. It parses changelog files (plain text or Markdown), creates embeddings, and returns highly relevant results ranked by relevance with rich version metadata. This enables AI agents to intelligently reason about system changes, architectural decisions, and historical context.

This server provides an intelligent layer over standard changelogs by combining vector search with BM25 keyword search, powered by Reciprocal Rank Fusion (RRF). It explicitly understands changelog structures (What, Why, Where, When) to answer both "what changed" and "why it changed".

## Key Features

- **Structure-Aware Chunking:** Parses Markdown tables and sections intelligently, preserving metadata instead of naively splitting text.
- **Hybrid Retrieval System:** Combines vector embeddings (`all-MiniLM-L6-v2`) with keyword search (`rank-bm25`) for superior retrieval accuracy.
- **Persistent Local Index:** Uses ChromaDB embedded mode to persist the index, ensuring fast startup times without re-indexing.
- **Incremental Indexing:** Detects file changes via SHA-256 hashes to only re-index what has changed.
- **Graceful Degradation:** Fails over to keyword search if embeddings fail, and handles corrupted files without crashing.
- **Relevance Validation:** Filters out low-quality results dynamically to provide high-confidence context to AI agents.

## Architecture

The server is built with a clean, layered architecture:
- **MCP Protocol Layer:** Utilizes the official `mcp` Python SDK (stdio transport).
- **Tool Layer:** Exposes domain-specific search tools to the AI client.
- **Retrieval Layer:** Implements hybrid search (Vector + BM25) with RRF fusion.
- **Storage Layer:** Local ChromaDB (vectors) and in-memory BM25 index.
- **Indexing Layer:** Handles incremental parsing, structure-aware chunking, and embedding.

## Requirements

- Python 3.11+
- [MCP CLI](https://github.com/modelcontextprotocol/cli) (optional, for development)

## Getting Started

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Run tests:**
   ```bash
   pytest tests/
   ```

3. **Run the MCP Server:**
   You can run the server directly using FastMCP's built-in support: 

   ```bash  
   # Run via MCP CLI Inspector (GUI) - WINDOWS
     .\venv\Scripts\Activate.ps1
     mcp dev src/server.py
   # if wsl/linux/mac
     source venv/bin/activate
     mcp dev src.server.py
   ```
   

   **Connecting via an MCP Client (GUI):**
   When configuring your MCP Client (like Claude Desktop or any other GUI client), you will need to set up the connection parameters:
   1. Enter the command as `python`
   2. Enter the argument as `-m src.server`
   3. Click connect.
   
   Once successfully connected, navigate to the **Tools** tab/section in the GUI, where you can write what you want to search for.

   **Proper Verified Examples for Tools:**
   
   - **`search_changelog`**
     - *Query*: "Fix Geofence duplicate on retry" (Should find entry #600 about adding a unique constraint)
     - *Query*: "What security changes were made to the reporting or tenancy modules?" (Should find entries #596, #593 about input size caps)
     - *Query*: "How was the N+1 issue fixed for Webhooks?" (Should find entry #576 about switching to `leftJoinAndSelect`)
   
   - **`search_decision_rationale`**
     - *Query*: "Why did login token revocation become instant?" (Will explain the move to Redis for immediate nbf epoch cutoff)
     - *Query*: "Why was TypeORM chosen over Prisma?" (Will detail the team choice and migration reasoning)
     - *Query*: "Why are payments idempotency-keyed?" (Will explain the prevention of double-charges on retries)

### Docker Deployment

A multi-stage Dockerfile is included for easy deployment.

1. **Build the image:**
   ```bash
   docker build -t changelog-mcp .
   ```

2. **Run the container:**
   You must mount your changelog directory so the server can index it, and optionally mount a data directory to persist the vector database across restarts.
   
   ```bash
   docker run -i \
     -v /path/to/your/changelogs:/changelogs \
     -v changelog-mcp-data:/data \
     changelog-mcp
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHANGELOG_DIR` | `./` | Directory containing markdown files to index. |
| `MCP_DATA_DIR` | `./data` | Directory to persist ChromaDB and BM25 index. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `RELEVANCE_HARD_FLOOR` | `0.1` | Minimum RRF score for results to be returned. |

## Available Tools

Once connected to an MCP client, the following tools are exposed:

- **`search_changelog(query, module?, limit?)`**: Search standard changelog entries for specific changes, features, or fixes.
- **`search_decision_rationale(query, limit?)`**: Search for architectural context and "Key decisions". Focuses heavily on the "Why" behind the changes.
- **`get_index_status()`**: Get current indexing status, last index time, database size, and source files.

## AI Disclosure

Artificial Intelligence (AI) tools were utilized during the development of this project to assist with generating boilerplate code, refining code structure and comments, cleaning up syntax, formatting Git commit messages, and aiding in debugging. 

However, all core architectural planning, system design decisions, tool concept development, and the implementation logic of the various components were strictly human-driven. The AI acted purely as a supportive assistant to accelerate development, while the critical thinking and engineering tradeoffs remained entirely in human hands.