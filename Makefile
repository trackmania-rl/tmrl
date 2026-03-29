.PHONY: fmt lint types check install-dev kill-server server trainer worker

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

install-dev:
	uv sync --group dev

kill-server:
ifeq ($(OS),Windows_NT)
	@echo Checking for processes on port 55555...
	@-powershell.exe -NoProfile -NonInteractive -Command 'Get-NetTCPConnection -LocalPort 55555 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $$_ -Force -ErrorAction SilentlyContinue }'
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
