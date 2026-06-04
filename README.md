# Seshat

Seshat is a GenAI pipeline that turns meeting recordings into a structured knowledge base. It ingests audio or pre-formatted transcripts, extracts Architecture Decision Records, risks, agreements, and action items, and writes them to a queryable store that tracks relationships between decisions across meetings — supersessions, amendments, and conflicts.

Built as a master's thesis project.

## Documentation map

- `docs/primer.md` → Developer primer: narrative overview and end-to-end job walkthrough.
- `docs/architecture.md` → Architecture summary: key design decisions and rationale.
- `docs/seshat-sdd.md` → Solution Design Document: implementation-oriented system design.
- `docs/superpowers/specs/2026-04-21-seshat-design.md` → Full design spec and detailed contracts.
- `docs/superpowers/specs/2026-04-24-quality-gate-design.md` → Quality gate design spec (pre-commit + GHA).
- `docs/superpowers/plans/2026-04-24-quality-gate.md` → Quality gate implementation plan.
- `docs/superpowers/specs/2026-04-27-prompt-interaction-design.md` → Prompt and interaction design spec.

## Project structure

```
seshat/
├── src/
│   └── seshat/
│       ├── agents/          # LLM agents: identification (extraction) and resolution families
│       ├── blob_store/      # S3 blob store abstraction (aioboto3)
│       ├── config/          # Pydantic settings (EvalConfig, LLMConfig, ConfidenceWeights, …)
│       ├── eval/            # MLflow-backed eval harnesses and calibration meta-scorers
│       ├── knowledge_store/ # Postgres-backed KB node persistence
│       ├── models/          # Pydantic domain models (KBNode, enums, …)
│       ├── observability/   # MLflow tracing and run management
│       ├── pipeline/        # ExtractionOrchestrator and extraction sub-pipeline
│       ├── secrets/         # AWS Secrets Manager helpers
│       ├── utils/           # Shared utilities
│       └── vector_store/    # pgvector semantic search abstraction
├── tests/
├── data/
├── docs/
├── alembic/
├── docker/
└── development/
```

## Running tests

The default `uv run pytest` run excludes the `llm` marker (see `addopts` in `pyproject.toml`). Use these commands depending on what you want to cover:

| Command | What runs |
|---------|-----------|
| `uv run pytest` | Default (excludes llm) |
| `uv run pytest -m ""` | All tests |
| `uv run pytest -m "not integration"` | Pure unit tests |
| `uv run pytest -m integration` | Integration tests (includes llm) |
| `uv run pytest -m "integration and not llm"` | Non-LLM integration (Postgres, LocalStack, MLflow) |
| `uv run pytest -m llm` | Tests requiring a live LLM API key |
| `uv run pytest -m agents` | Agent tests (subset of llm) |
| `uv run pytest -m embedding` | Embedding tests (subset of llm) |

**LLM tests cost money.** Always redirect output to a file when running them:

```bash
uv run pytest -m llm 2>&1 | tee /tmp/llm_test_out.txt
```

## Common uv commands

| Command | Description |
|---------|-------------|
| `uv run pytest` | Run tests (according to the default addopts defined in `pyproject.toml`) |
| `uv run ruff check src` | Lint |
| `uv run ruff format src` | Format |
| `uv run mypy` | Type check |
| `uv run radon cc src -a -nb` | Cyclomatic complexity (shows B+ rated items only) |
| `uv sync --group eval` | Install optional eval dependencies |
