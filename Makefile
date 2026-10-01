.PHONY: install up down test lint fmt type case hello eval deploy clean

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

eval:
	uv run alibi eval --games 20 --gate

deploy:
	docker compose -f compose.homelab.yml up -d --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
