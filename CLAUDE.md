# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral rules

## The Four Principles in Detail

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

LLMs often pick an interpretation silently and run with it. This principle forces explicit reasoning:

- **State assumptions explicitly** — If uncertain, ask rather than guess
- **Present multiple interpretations** — Don't pick silently when ambiguity exists
- **Push back when warranted** — If a simpler approach exists, say so
- **Stop when confused** — Name what's unclear and ask for clarification

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

Combat the tendency toward overengineering:

- No features beyond what was asked
- No abstractions for single-use code
- No "flexibility" or "configurability" that wasn't requested
- No error handling for impossible scenarios
- If 200 lines could be 50, rewrite it

**The test:** Would a senior engineer say this is overcomplicated? If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style, even if you'd do it differently
- If you notice unrelated dead code, mention it — don't delete it

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused
- Don't remove pre-existing dead code unless asked

**The test:** Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform imperative tasks into verifiable goals:

| Instead of... | Transform to... |
|--------------|-----------------|
| "Add validation" | "Write tests for invalid inputs, then make them pass" |
| "Fix the bug" | "Write a test that reproduces it, then make it pass" |
| "Refactor X" | "Ensure tests pass before and after" |

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let the LLM loop independently. Weak criteria ("make it work") require constant clarification.

## Project Structure

```
seshat/
├── src/seshat/
│   ├── agents/          # LLM agents: identification (extraction) and resolution families
│   ├── blob_store/      # S3 blob store abstraction (aioboto3)
│   ├── config/          # Pydantic settings (EvalConfig, LLMConfig, ConfidenceWeights, …)
│   ├── eval/            # MLflow-backed eval harnesses and calibration meta-scorers
│   ├── knowledge_store/ # Postgres-backed KB node persistence
│   ├── models/          # Pydantic domain models (KBNode, enums, …)
│   ├── observability/   # MLflow tracing and run management
│   ├── pipeline/        # ExtractionOrchestrator and extraction sub-pipeline
│   ├── secrets/         # AWS Secrets Manager helpers
│   ├── utils/           # Shared utilities
│   └── vector_store/    # pgvector semantic search abstraction
├── tests/               # pytest test suite (unit/ and integration/)
├── data/eval/           # Ground-truth YAML corpus fixtures for the eval harnesses
├── alembic/             # DB migration scripts
├── docs/                # Architecture docs, SDD, design specs
└── pyproject.toml       # Single source of truth for deps, tool config, metadata
```
## Running Tests

The default `uv run pytest` run excludes the `llm` marker (see `addopts` in `pyproject.toml`). Use these commands depending on what you want to cover:

| Command | What runs |
|---------|-----------|
| `uv run pytest` | Default behavior, as defined in `addopts` |
| `uv run pytest -m ""` | All tests (unit and integration) |
| `uv run pytest -m "not integration"` | Pure unit tests only |
| `uv run pytest -m integration` | Integration tests only including, `llm` ones |
| `uv run pytest -m "integration and not llm"` | Non-LLM integration tests only (Postgres, LocalStack, MLflow) |
| `uv run pytest -m llm` | All tests needing a live LLM API key |
| `uv run pytest -m agents` | Agent tests (subset of `llm`) |
| `uv run pytest -m embedding` | Embedding tests (subset of `llm`) |

Markers are defined in `pyproject.toml` under `[tool.pytest.ini_options]`. The `llm` and `agents` markers also require the relevant API keys to be present in the environment.

**LLM tests cost money.** When running any test with the `llm`, `agents`, or `embedding` markers, always redirect output to a temporary file so the results can be reviewed without re-running:

```bash
uv run pytest -m llm 2>&1 | tee /tmp/llm_test_out.txt
```

## Test Style Guide

- **Classes**: use a test class per target class, with one test method per method under test.
- **Functions**: use top-level test functions (one or more per target function).
- Avoid redundant or shallow tests, e.g., check config defaults or `auto()` enum values.

## Code Style

- Add a blank line after closing `with` blocks before the next statement.
- Add a blank line after `if` / `try-except` blocks before the next statement, but not before the block itself.
- Use match case over multiple if blocks for factory methods
- Use logs via stdlib `logging` over bare print statements

## Completing Implementation Tasks

When finishing any implementation task, create a TODO list with these steps:
1. Use the `superpowers:finishing-a-development-branch` skill to guide integration.
2. Check whether `docs/seshat-sdd.md` or the relevant spec under `docs/superpowers/specs/` needs updating to reflect any design decisions that changed during implementation.

## Package Manager

This project uses `uv`. All Python commands must run through `uv`, e.g. `uv run pytest`, `uv run python`, `uv add <package>`, `uv pip show <package>`.

## Spec Compliance

`docs/superpowers/specs/2026-04-21-seshat-design.md` is the project source of truth. When executing an implementation plan, if the plan conflicts with the spec, follow the spec and deviate from the plan so the resulting code matches it.

The only exception is when the user explicitly asks to deviate from the spec — in that case, update the spec after the implementation task is done to reflect the deviation.

## Notes

- Architecture and AI component decisions are documented in `docs/architecture.md`, `docs/seshat-sdd.md`, and the specs under `docs/superpowers/specs/`.
- `.claude/claude-behavior.md` is a personal Claude config file, not project documentation — ignore it.
- `pyproject.toml` is the single source of truth for dependencies, tool config (ruff, pytest), and project metadata.
