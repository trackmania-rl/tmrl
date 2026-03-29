.PHONY: fmt lint types check test tests install-dev kill-server server trainer worker record-episode

# Unix: .venv | Windows: .venv-windows (override with e.g. UV_ENV=.venv-other make server)
UV_ENV ?= $(if $(filter Windows_NT,$(OS)),.venv-windows,.venv)
UV_TORCH_BACKEND ?= auto
export UV_TORCH_BACKEND

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
ifeq ($(OS),Windows_NT)
	@echo Checking for processes on port 55555...
	@for /f "tokens=5" %%a in ('netstat -aon ^| findstr :55555') do @taskkill /F /PID %%a 2>nul
	@exit 0
else
	@echo "Checking for processes on port 55555..."
	@-command -v fuser >/dev/null 2>&1 && fuser -k 55555/tcp 2>/dev/null || true
	@-lsof -ti:55555 2>/dev/null | xargs -r kill -9 2>/dev/null || true
endif

server: kill-server
ifeq ($(OS),Windows_NT)
	@set "UV_PROJECT_ENVIRONMENT=$(UV_ENV)" && uv run python -m tmrl --server
else
	@UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run python -m tmrl --server
endif

trainer:
ifeq ($(OS),Windows_NT)
	@set "UV_PROJECT_ENVIRONMENT=$(UV_ENV)" && uv run python -m tmrl --trainer
else
	@UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run python -m tmrl --trainer
endif

worker:
ifeq ($(OS),Windows_NT)
	@set "UV_PROJECT_ENVIRONMENT=$(UV_ENV)" && uv run python -m tmrl --worker
else
	@UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run python -m tmrl --worker
endif

record-episode:
ifeq ($(OS),Windows_NT)
	@set "UV_PROJECT_ENVIRONMENT=$(UV_ENV)" && uv run python -m tmrl --record-episode --record-episode-count $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),2)
else
	@UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run python -m tmrl --record-episode --record-episode-count $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),2)
endif

# Allow: make record-episode 5
%:
	@:
