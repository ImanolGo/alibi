.PHONY: install up down test lint fmt type case hello clean

install:
	uv sync

up: ## start postgres + phoenix
	docker compose up -d

down:
	docker compose down

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff check --fix .
	uv run ruff format .

case:
	uv run alibi case --seed 42

hello:
	uv run alibi hello

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
