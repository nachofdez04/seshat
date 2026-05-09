# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is an MVP — structure will be updated as components are defined. Expected layout:

```
src/          # Application source code
tests/        # pytest test suite
pyproject.toml
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

## Notes

- Architecture and AI component decisions are documented in `docs/architecture.md`, `docs/seshat-sdd.md`, and the specs under `docs/superpowers/specs/`.
- `.claude/claude-behavior.md` is a personal Claude config file, not project documentation — ignore it.
- `pyproject.toml` is the single source of truth for dependencies, tool config (ruff, pytest), and project metadata.
