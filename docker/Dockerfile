FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Install dependencies
COPY pyproject.toml uv.lock /app/
# Sync dependencies only first
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY common /app/common
COPY services /app/services
COPY README.md /app/

# Install project
RUN uv sync --frozen --no-dev

# Add venv to path
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["uvicorn", "--help"]
