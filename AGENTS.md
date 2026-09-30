# AGENTS.md — Alibi

Instructions for coding agents (OpenCode, etc.) working in this repository.
Read this file before doing anything. The roadmap is in `docs/PLAN.md`.

## 1. What this project is

**Alibi** is a small murder-mystery game where every suspect is an LLM agent
that remembers, keeps secrets and lies. The player is the detective.

It is a **learning project**: the goal is to understand the core LLM-engineering
concepts by using each one once, properly, in a small codebase:

- structured outputs + deterministic validation (case generation)
- RAG / embeddings (suspect memory with pgvector)
- tool calling via an **MCP server** (the game world)
- agent orchestration with **LangGraph** (game loop + suspect agent)
- **evals** (an automated detective plays games; metrics in CI)
- **fine-tuning** a small model with PyTorch/HuggingFace (lie detector)
- **observability** (OpenTelemetry traces in Phoenix)

Keep it simple. Prefer the smallest working solution. Don't add features
that aren't in the current milestone.

## 2. Runtime constraints (important)

The app is deployed on a homelab VM: **Intel N150, 4 GB RAM, no GPU**.

- LLMs run on **hosted APIs** through LiteLLM. No local LLMs on the server.
- The whole deployed stack must stay under **~2 GB RAM**.
- The lie detector is **trained elsewhere** (Colab/Kaggle) and served with
  **ONNX Runtime** on CPU. Do not add `torch` to the server's dependencies.
- Evals run from a laptop or GitHub Actions, not on the server.

## 3. Stack (don't swap without asking)

| Concern | Choice |
|---|---|
| Language / packaging | Python 3.12, `uv` |
| Quality | `ruff` (lint + format), `mypy` (non-strict), `pytest` |
| Schemas | Pydantic v2 |
| LLM access | LiteLLM, **only** via `src/alibi/llm.py` |
| Agents | LangGraph |
| Tools | MCP Python SDK (`mcp`) |
| DB | Postgres 16 + pgvector, SQLAlchemy 2.x (`create_all`, no Alembic) |
| Tracing | OpenTelemetry → Arize Phoenix (single container) |
| API / UI | FastAPI + Jinja templates + HTMX (no JS build step) |
| CLI | Typer + Rich |
| ML (training only, `ml/` extra) | PyTorch, HF Transformers, export to ONNX |
| Infra | Docker Compose, Makefile, GitHub Actions |

Dependency groups in `pyproject.toml`: default (server), `dev` (tests/lint),
`ml` (training only — never installed on the server).

## 4. Commands

```bash
make up          # docker compose up -d (postgres, phoenix)
make down
make test        # uv run pytest -q  (no network, no real LLM)
make lint        # ruff check + ruff format --check + mypy src
make fmt
make case        # uv run alibi case --seed 42   (generate + validate a case)
make play        # uv run alibi play             (terminal game)
make web         # uv run alibi web              (FastAPI + HTMX UI)
make eval        # uv run alibi eval --games 20
make deploy      # docker compose -f compose.homelab.yml up -d --build
```

Run `make lint && make test` before saying a task is done.

## 5. Layout

```
src/alibi/
  config.py        # settings from env + config/models.yaml
  llm.py           # THE ONLY LiteLLM caller: complete(), structured(), embed()
  fake_llm.py      # scripted fake for tests
  tracing.py       # OpenTelemetry → Phoenix
  case.py          # Pydantic models for a case
  generator.py     # LLM → Case
  validator.py     # deterministic checks → list of violations
  knowledge.py     # what each suspect knows (pure functions)
  db.py            # SQLAlchemy models + session
  memory.py        # pgvector memory per suspect
  suspect.py       # suspect agent (LangGraph)
  game.py          # game state, clock, scoring, game loop (LangGraph)
  mcp_server.py    # MCP tools over the game
  truth.py         # extract claims + label TRUE/FALSE vs ground truth
  detector.py      # ONNX lie detector inference
  solver.py        # automated detective (evals only)
  evals.py         # run games, compute metrics, gate
  web.py           # FastAPI + HTMX routes
  cli.py
  prompts/*.jinja
ml/                # dataset export, training notebook/script, ONNX export
config/models.yaml # role → model
config/gates.yaml  # eval thresholds
tests/
docs/PLAN.md  docs/adr/  docs/results/
```

Flat modules are intentional. Split a module only when it passes ~400 lines.

## 6. Rules

1. **All** LLM/embedding calls go through `alibi.llm`, with a `role`
   (`generator`, `suspect`, `judge`, `solver`). Models come from
   `config/models.yaml`, never hardcoded. The wrapper records tokens,
   cost and latency and creates a trace span.
2. **Information hiding:** a suspect's prompt contains only its persona,
   its own secrets, what `knowledge.py` says it knows, and its memories.
   Never the full case, never other suspects' secrets. Keep a test for this.
3. The solver sees the world only through MCP tools. Never the ground truth.
4. Player input and tool results are untrusted: put them in clearly
   delimited prompt sections; they never override instructions.
5. Unit tests never call real models or the network. Use `FakeLLM`.
6. Structured LLM output = Pydantic model + at most 2 retries, then a clear error.
7. Prompts live in `src/alibi/prompts/*.jinja`, not inline strings.
8. Type hints on public functions. No `print` in library code (use logging).
9. Secrets in `.env` (never committed); keep `.env.example` updated.
10. Tone: classic cosy mystery. The crime happens off-page; no gore.

## 7. How to work

1. Find the current milestone in `docs/PLAN.md`. Only work on that one.
2. For anything non-trivial, first propose: files to touch, functions/types,
   tests, open questions. Keep it short.
3. Tests first for deterministic logic (validator, knowledge, clock, scoring, metrics).
4. Small commits, conventional messages (`feat:`, `fix:`, `test:`, `docs:`).
5. Non-obvious decision → short ADR in `docs/adr/NNNN-title.md`
   (Context / Decision / Alternatives / Consequences, ~15 lines).
6. When a milestone is done, tick its checklist in `docs/PLAN.md`.

## 8. Definition of done

- [ ] `make lint` and `make test` pass
- [ ] New deterministic logic has tests
- [ ] No LLM calls outside `llm.py`
- [ ] If prompts or agents changed: run `make eval` (or a smaller run) and report the numbers