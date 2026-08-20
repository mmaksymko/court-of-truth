#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev --group training
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy src
uv run --no-sync pytest
