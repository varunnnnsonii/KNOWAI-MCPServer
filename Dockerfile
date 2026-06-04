# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml .
# Create dummy src so pip can install the project
RUN mkdir -p src && touch src/__init__.py

RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels . && \
    pip install --no-cache-dir --target=/app/deps ".[cli]"


# Stage 2: Final runtime
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies for rank-bm25/chroma/etc if needed
# (Chroma usually requires sqlite3 which is built into python-slim)

# Copy dependencies from builder
COPY --from=builder /app/deps /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/mcp /usr/local/bin/mcp

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Environment configuration
ENV PYTHONPATH=/app
ENV MCP_DATA_DIR=/data
ENV CHANGELOG_DIR=/changelogs
ENV LOG_LEVEL=INFO

# Create necessary directories
RUN mkdir -p /data /changelogs

# Expose standard stdio
# FastMCP uses stdio by default when using `mcp run` or `python src/server.py`
# But if you want to run an SSE server, it's also possible. We default to stdio.

ENTRYPOINT ["python", "-m", "src.server"]
