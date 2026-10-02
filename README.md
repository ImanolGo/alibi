![Alibi — a murder mystery where every suspect is an LLM](assets/alibi-banner.jpeg)

# Alibi

A small murder mystery where every suspect is an LLM agent that remembers, keeps
secrets and lies. You are the detective.

Alibi is a **learning project**: one clean, working example of each core
LLM-engineering concept (structured outputs, RAG, MCP, agent orchestration,
evals, fine-tuning, observability) in a codebase small enough to read. The
roadmap and milestone checklists live in [`docs/PLAN.md`](docs/PLAN.md).

## Status

| Milestone | State |
|---|---|
| M0 Skeleton | ✅ |
| M1 The Case File | 🚧 (10/10 cases generate; paper-solving pending) |
| M2 One Suspect | ✅ |
| M3 World + MCP | ✅ |
| M4 Evals | 🚧 (code + gates done; full 20-game numbers pending) |
| M5 Detector + Web + Homelab | 🚧 (web UI done; deploy started; detector + training pending) |

## Quickstart

```bash
make install                 # uv sync
make up                      # postgres + phoenix (docker)
cp .env.example .env         # add your OPENROUTER_API_KEY
make hello                   # one real LLM call: tokens, cost, latency (traced)

make case                    # generate + validate a case into cases/
uv run alibi interrogate --case cases/<id>.json --suspect <id> --debug
uv run alibi play --case cases/<id>.json --new
make test && make lint
```

## Architecture

```
            Browser (HTMX)      Terminal (Rich)     Any MCP client
                  │                   │                   │
                  └─────── FastAPI / CLI ──────┐          │
                                               ▼          ▼
                                     MCP server (game tools)
                                               │
                              Game loop (LangGraph): actions, turns, score
                                               │
                       ┌───────────────────────┼─────────────────────┐
                       ▼                       ▼                     ▼
               Suspect agent (LangGraph)   Truth labeller      Lie detector
               retrieve → think → speak    claims → TRUE/FALSE (ONNX, CPU, M5)
               → remember                        │
                       │                         ▼
                 pgvector memory          stored labels → evals + training data
                       │
                       ▼
     llm.py (LiteLLM) ── hosted models, cost/latency/tokens, daily spend cap
                       │
     OpenTelemetry ── Phoenix

     Offline (laptop / CI): case generator + validator, solver agent, evals,
                            lie-detector training (Colab) → model.onnx
```

## The stack, and why

Every choice is here to teach one thing. The "why" is the point.

| Concern | Choice | Why |
|---|---|---|
| Language / packaging | **Python 3.12 + `uv`** | `uv` replaces pip/venv/pyenv wheel; one lockfile, fast installs, reproducible builds. |
| Schemas | **Pydantic v2** | A schema *is* the contract for structured LLM output; parsing and validation come for free. |
| LLM access | **LiteLLM**, behind `llm.py` | One interface over many providers (`openrouter/…`, `openai/…`); swap models in `config/models.yaml`, no code change. Also gives us token/cost accounting. |
| Agent orchestration | **LangGraph** | Agents as explicit state machines (`retrieve → think → speak → remember`). Edges and nodes make the loop inspectable and testable, unlike a hidden ReAct string loop. |
| Tools | **MCP Python SDK** | A standard tool protocol: the game is exposed once and used by the terminal, the solver, and external clients alike. Tool descriptions double as prompts. |
| Memory | **Postgres 16 + pgvector** | One database for both durable game state *and* vector search — no separate vector store to run on a 4 GB box. |
| DB access | **SQLAlchemy 2.x**, `create_all` | Typed models and queries; migrations (Alembic) are overkill for a project this size. |
| Observability | **OpenTelemetry → Arize Phoenix** | Vendor-neutral spans for every node and LLM call, with cost/latency attached. Tracing is wired in from M0 and used everywhere. |
| API / UI | **FastAPI + Jinja + HTMX** | Server-rendered pages, no JS build step (M5). |
| CLI | **Typer + Rich** | Type-hint-driven commands; readable terminal output. |
| ML (training only) | **PyTorch + HF**, export to **ONNX** | Training happens on Colab/Kaggle; the server only runs ONNX Runtime on CPU — so `torch` never touches the homelab. |
| Infra | **Docker Compose + Makefile + GitHub Actions** | Same `compose` for dev and homelab; CI runs lint, tests and (for prompt changes) an eval gate. |

## How it works

Each milestone adds one concept; see `docs/results/` for the measured outcomes.

1. **M1 — structured output + validation.** One LLM call fills a Pydantic `Case`
   from `config/setting.yaml`; a deterministic validator checks six rules and a
   bounded repair loop feeds violations back to the model. Creativity is the
   model's job; correctness is the code's. (`case.py`, `generator.py`,
   `validator.py`)
2. **M2 — RAG memory + a suspect agent.** Each suspect stores memories in
   pgvector and answers through a LangGraph. Information hiding is a code
   guarantee: the prompt only ever sees the persona, the suspect's own secrets
   and what `knowledge.py` says they know. (`knowledge.py`, `memory.py`,
   `suspect.py`)
3. **M3 — the world and MCP.** A `Game` owns the rules (20 actions, scoring,
   persistence); the terminal and an MCP server are thin adapters over it. State
   lives in Postgres, so a game survives a restart. (`game.py`, `mcp_server.py`)
4. **M4 — evals.** An automatic detective plays through the tools; a judge
   labels every claim against the ground truth; metrics gate changes in CI.
   (`solver.py`, `truth.py`, `evals.py`, `config/gates.yaml`)
5. **M5 — detector, web UI, homelab.** A tiny DistilBERT fine-tuned on our own
   TRUE/FALSE claims, exported to ONNX, plus a browser UI and the deployment
   below.

## Deploy (Docker / homelab)

The app is one small image (`python:3.12-slim`, no torch) served as MCP over
streamable HTTP, alongside Postgres and Phoenix — sized to stay under ~2 GB on
an Intel N150 with no GPU.

```bash
cp .env.example .env                      # set OPENROUTER_API_KEY (+ low budget)
docker compose -f compose.homelab.yml run --rm app alibi case --seed 1
make deploy                               # docker compose -f compose.homelab.yml up -d --build
```

Or build/run the image directly:

```bash
docker build -t alibi .
docker run --rm -p 8080:8080 --env-file .env -v alibi_cases:/app/cases alibi
```

Full walkthrough — Cloudflare Tunnel (public access, no open ports), cost caps,
memory limits and Postgres backup — is in
[`docs/deploy-homelab.md`](docs/deploy-homelab.md).

## Play from an MCP client

The whole game is exposed as MCP tools. Point any client at the stdio server:

```json
{
  "mcpServers": {
    "alibi": {
      "command": "uv",
      "args": ["run", "alibi", "mcp", "--case", "cases/<id>.json"],
      "cwd": "/absolute/path/to/alibi"
    }
  }
}
```

Or at the deployed HTTP endpoint: `{ "type": "http", "url": "http://<vm>:8080/mcp" }`.

Tools: `case_summary`, `status`, `list_rooms`, `search_room`, `inspect`,
`question`, `lie_detector` (stub until M5), `accuse`.

## Play in the browser

```bash
uv run alibi web        # http://127.0.0.1:8000
```

FastAPI + Jinja + HTMX, no JS build step. Generate a case (or pick one), search
rooms, inspect the evidence, question suspects — presenting evidence when you
have it — then accuse, and read "what really happened". The same `Game` object
backs the web, CLI and MCP server, so a game is shared across all three.

## Evals

```bash
make eval                          # uv run alibi eval --games 20 --gate
uv run alibi eval --games 5 --seed 1
```

Metrics (`config/gates.yaml`): `case_valid_rate`, `solve_rate`,
`contradiction_rate`, `early_confession_rate`, `cost_per_game`. A GitHub Action
runs a 5-game gate on PRs touching `src/alibi/prompts/**` or `suspect.py` and
comments the summary on the PR. The gate is exercised by deliberately breaking a
suspect prompt: _<link to the closed "broken prompt" PR here>_.

## Layout

```
src/alibi/        case · generator · validator · knowledge · llm · memory
                  suspect · game · mcp_server · solver · truth · evals · cli
config/           models.yaml (role → model) · setting.yaml · gates.yaml
tests/            deterministic tests only (no network; FakeLLM)
ml/               dataset export + training (M5; never installed on the server)
assets/           the banner
docs/PLAN.md      roadmap · docs/adr/ decisions · docs/results/ measurements
Dockerfile        small runtime image        compose.homelab.yml  the deploy
```

## Rules of the house

- Every LLM/embedding call goes through `src/alibi/llm.py`, with a role.
- Models come from `config/models.yaml`; prompts from `src/alibi/prompts/*.jinja`.
- The validator, not the model, decides whether a case is sound.

## License

MIT — see [`LICENSE`](LICENSE).
