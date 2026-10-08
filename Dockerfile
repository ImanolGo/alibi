# Small runtime image for the N150 homelab (no torch, no dev deps).
# Build:  docker build -t alibi .
# Run:    docker run --rm -p 8080:8080 --env-file .env -v alibi_cases:/app/cases alibi

# --- build -----------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependencies first (better layer caching).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    ALIBI_ROOT=/app
WORKDIR /app

COPY --from=build /app /app
COPY config ./config
RUN mkdir -p /app/cases

EXPOSE 8080

# Serves the game over MCP by default while the app/web service serves the UI.
CMD ["alibi", "mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8080"]
