# Repository Guidelines

This project contains Python utilities for orchestrating multi-agent workflows for knowledge graph enrichment. The code is primarily in Python 3.9+ and makes heavy use of asynchronous functions and Pydantic models.

## Coding Standards

- **Language**: Python 3.9 or newer.
- **Formatting**: run `black .` before committing.
- **Linting**: run `ruff .` to check code style.
- **Type Checking**: run `mypy .` to ensure type hints are valid.
- **Docstrings**: use triple double quotes and include argument and return descriptions.
- **Asynchronous Design**: many workflow functions are `async`—follow existing patterns when adding new steps or agents.

## Repository Structure

- `main.py` – CLI entry point to run the workflow.
- `orchestrator.py` – coordinates agents and workflow steps.
- `workflow_agents.py` – defines agent instances.
- `steps/` – individual step implementations for the workflow.
- `config.py` – configuration constants and environment variable loading.
- `schemas.py` – Pydantic models defining expected data structures.
- `utils.py` – helper utilities (logging, I/O, agent runner, etc.).

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) for clarity:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation updates
- `chore:` for maintenance tasks
- `refactor:` for code restructuring

## Pull Requests

When submitting a PR, include a short summary of what changed and mention any relevant issue numbers. Ensure that formatting, linting and type checks all pass before opening the PR.

