# Alibi

A small murder mystery where every suspect is an LLM agent that remembers,
keeps secrets and lies. You are the detective.

Learning project: one clean example of each core LLM-engineering concept
(structured outputs, RAG, MCP, LangGraph, evals, fine-tuning, observability) in
a tiny codebase. See `docs/PLAN.md` for the roadmap.

## Status

M0 (skeleton) and M1 (case file) — in progress.

## Quickstart

```bash
make install          # uv sync
make up               # postgres + phoenix (docker)
make test             # unit tests, no network
make lint

cp .env.example .env  # add your provider API key
make hello            # one real LLM call: answer, tokens, cost, latency
make case             # generate + validate a case, save to cases/
uv run alibi interrogate --case cases/<id>.json --suspect <suspect_id> --debug
```

## Layout

```
src/alibi/        case · generator · validator · knowledge · llm · cli
config/           models.yaml (role -> model) · setting.yaml
tests/            deterministic tests only (no network, FakeLLM)
ml/               dataset export + training (M5, never installed on the server)
docs/PLAN.md      roadmap · docs/adr/ decisions · docs/results/ measurements
```

## Rules of the house

- Every LLM/embedding call goes through `src/alibi/llm.py`, with a role.
- Models come from `config/models.yaml`; prompts from `src/alibi/prompts/*.jinja`.
- The validator, not the model, decides whether a case is sound.
