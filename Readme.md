this is a public repo linked/should be linked to https://github.com/varunnnnsonii/KNOWAI-MCPServer

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
   # Run via MCP CLI Inspector (GUI)
   mcp dev src.server:mcp

   # Run directly via stdio (for MCP Clients like Claude Desktop)
   python -m src.server
   ```

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

## Tool Usage

Once connected to an MCP client, the following tools are available:

- `search_changelog(query, module?, limit?)`: Search standard changelog entries.
- `search_decision_rationale(query, limit?)`: Search for architectural context and "Key decisions".
- `get_index_status()`: Get current indexing status and database size.