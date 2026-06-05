# Stage 1: Build environment
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pyproject.toml
COPY pyproject.toml .

# Create dummy src so pip can install the project without needing the full code yet
RUN mkdir -p src && touch src/__init__.py

# Install dependencies into the virtual environment
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .[cli]

# Stage 2: Final runtime
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies if needed
# (Chroma usually requires sqlite3 which is built into python-slim)

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Ensure the virtual environment is used
ENV PATH="/opt/venv/bin:$PATH"

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

# FastMCP uses stdio by default when running directly
ENTRYPOINT ["python", "-m", "src.server"]
