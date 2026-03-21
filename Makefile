.PHONY: fmt lint types check test tests install-dev server trainer worker record-episode

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

types:
	uv run mypy tmrl

check: lint types

test:
	uv run pytest

tests:
	uv run pytest tests/ -v

install-dev:
	uv sync --group dev

kill-server:
	@echo "Checking for zombie processes on port 55555..."
	@for /f "tokens=5" %%a in ('netstat -aon ^| findstr :55555') do taskkill /F /PID %%a 2>nul || exit 0

server: kill-server
	@set "UV_PROJECT_ENVIRONMENT=.venv-windows" && uv run python -m tmrl --server

trainer:
	@set "UV_PROJECT_ENVIRONMENT=.venv-windows" && uv run python -m tmrl --trainer

worker:
	@set "UV_PROJECT_ENVIRONMENT=.venv-windows" && uv run python -m tmrl --worker

record-episode:
	@set "UV_PROJECT_ENVIRONMENT=.venv-windows" && uv run python -m tmrl --record-episode --record-episode-count $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),2)

# Allow syntax like: make record-episode 5
%:
	@:
