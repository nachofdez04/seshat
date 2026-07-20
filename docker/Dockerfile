FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS deps
WORKDIR /app
ENV PYTHONPATH=/app/src
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev


FROM deps AS migrate
COPY alembic.ini .
COPY alembic/ alembic/
COPY src/ src/
CMD ["uv", "run", "--no-sync", "seshat", "migrate"]


FROM deps AS api
COPY src/ src/
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "seshat", "api"]
