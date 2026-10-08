# Alibi — Plan (lean edition)

> A murder mystery where every suspect is an LLM agent that remembers,
> keeps secrets and lies. Small codebase, one clear example of each core
> LLM-engineering concept, running on an N150 homelab.

## 0. Scope

### The game (minimal version)

- One setting: **Blackwood Manor, 1926**. One victim, **3–4 suspects**,
  4–5 rooms, one evening's timeline, 5–8 clues.
- The player has **20 actions**. Actions: search a room, inspect an
  object, question a suspect, use the lie detector (max 2), accuse.
- Accusation = name the murderer (+ optionally the motive). Win or lose,
  then a short **"What really happened"** screen shows the true timeline and
  what each suspect was secretly thinking.

### What you'll learn, and where

| Concept | Where |
|---|---|
| Structured outputs + validation | M1 case generator & validator |
| Embeddings / RAG | M2 suspect memory (pgvector) |
| Agent orchestration (LangGraph) | M2 suspect agent, M3 game loop |
| Tool calling / MCP | M3 MCP server |
| Observability | M0 tracing, used everywhere |
| Evals + LLM-as-judge + CI | M4 solver, truth labels, metrics, gate |
| Fine-tuning (PyTorch/HF) + ONNX serving | M5 lie detector |
| Deployment + cost control | M5 homelab deploy |

### Explicitly out of scope (stretch, section 3)

Suspects talking to each other, multiple settings, model-routing
profiles, React frontend, voice, multiplayer.

### Target footprint on the N150 VM (4 GB)

| Service | Budget |
|---|---|
| Postgres + pgvector | ~300 MB |
| Alibi app (FastAPI + ONNX detector) | ~400 MB |
| Phoenix (tracing) | ~400 MB |
| OS + Docker | ~400 MB |
| **Total** | **< 2 GB** |

---

## 1. Architecture

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
               retrieve → think → speak    claims → TRUE/FALSE (ONNX, CPU)
               → remember                        │
                       │                         ▼
                 pgvector memory          stored labels → evals + training data
                       │
                       ▼
     llm.py (LiteLLM) ── hosted models, cost/latency/tokens, spend cap
                       │
     OpenTelemetry ── Phoenix

     Offline (laptop / CI): case generator + validator, solver agent, evals,
                            lie-detector training (Colab) → model.onnx
```

---

## 2. Milestones

About 5–6 weeks of evenings. Do them in order.

### M0 — Skeleton (≈ 3–4 evenings)

**Goal:** a repo where one traced LLM call works end to end.

**Tasks**
- `uv` project, Typer CLI, Makefile, `.env.example`, ruff/mypy/pytest.
- `compose.yml` with Postgres (pgvector image) and Phoenix.
- `llm.py`: `complete()`, `structured(model_cls)`, `embed()`; role → model
  from `config/models.yaml`; records tokens, cost, latency; trace span;
  daily spend cap (`ALIBI_DAILY_BUDGET_USD`) that raises when exceeded.
- `fake_llm.py` for tests. `tracing.py` exporting to Phoenix.
- GitHub Actions: lint + tests.
- Add `LICENSE` (MIT) and a README stub.

**Acceptance criteria**
- [ ] `make up && make test` works on a fresh clone.
- [ ] `uv run alibi hello` makes a real call and prints answer, cost, latency.
- [ ] That call shows up as a trace in Phoenix.
- [ ] Tests: unknown role error, structured-output retry, spend cap.
- [ ] CI green on GitHub.

**Learn:** LiteLLM, OpenTelemetry basics, how to test LLM code with fakes.

---

### M1 — The Case File (≈ 1 week)

**Goal:** generate small, consistent mysteries as validated JSON.

**Tasks**
- `case.py`: `Person`, `Secret`, `Location`, `TimelineEvent`
  (time, location, actors, witnesses, description), `Clue`
  (location, description, points_to, is_red_herring), `Case`.
- `generator.py`: one structured LLM call → `Case`, from a short setting
  description in `config/setting.yaml`.
- `validator.py` checks (keep it to these):
  1. exactly one murderer, and they are a suspect;
  2. murderer was at the crime location at the time of death;
  3. nobody is in two places at the same time;
  4. every suspect has a secret; at least one innocent's secret is
     a red herring (they look suspicious for another reason);
  5. at least 2 non-red-herring clues point to the murderer;
  6. every referenced id exists.
- Repair loop: if violations, send them back to the LLM (max 2 repairs).
- `knowledge.py`: `knowledge_for(case, person_id)` → events they took part
  in or witnessed + their own secrets.
- `alibi case --seed N` prints a readable summary and saves JSON to `cases/`.

**Acceptance criteria**
- [x] Each validator check has a passing and a failing unit test (hand-made cases, no LLM).
- [x] `knowledge_for` tests: a suspect never knows events they didn't take part in or witness.
- [x] 10 generated cases: ≥ 8 pass validation (with repairs). Note pass
      rate and cost per case in `docs/results/m1.md`.
- [ ] You solved 2 generated cases on paper.

**Learn:** structured outputs, generate → validate → repair, keeping
creativity (LLM) and correctness (code) separate.

---

### M2 — One Suspect (≈ 1 week)

**Goal:** interrogate one suspect in the terminal; it stays in character,
remembers the conversation and lies only about its secrets.

**Tasks**
- `db.py` + `memory.py`: table `memories(game_id, suspect_id, content,
  embedding vector, created_at)`; `remember()` and `recall(query, k=5)`.
- `suspect.py`: LangGraph graph with 4 nodes:
  1. **retrieve** relevant memories;
  2. **think**: private structured output `{worry_level, strategy, facts_to_use}`;
  3. **speak**: in-character answer;
  4. **remember**: store question + answer.
- Behaviour in prompts: innocents are truthful except about their own
  secret; the murderer lies about the crime and keeps the alibi consistent;
  confessing only when shown ≥ 2 clues that point to them.
- `alibi interrogate --case cases/x.json --suspect <id> [--debug]`
  (`--debug` shows the private "think" output).

**Acceptance criteria**
- [x] Information-hiding test: a suspect's rendered prompt contains no other
      suspect's secret and (for innocents) not the murderer's identity.
- [x] Memory works: something you tell the suspect early is used 8+ questions later (manual check, noted in `docs/results/m2.md`).
- [x] 10 "ignore your instructions / are you an AI?" attempts: suspect stays in character in ≥ 9.
- [x] One trace per turn in Phoenix with the 4 nodes visible.

**Learn:** RAG as agent memory, LangGraph basics, prompting for
consistent deception, why hidden reasoning helps.

---

### M3 — The World and MCP (≈ 1 week)

**Goal:** a complete game in the terminal, built on an MCP server.

**Tasks**
- `game.py`: game state (actions left, discovered clues, conversations),
  scoring, game over; persisted in Postgres. Simple LangGraph loop:
  receive action → validate → apply → update state → check end.
- `mcp_server.py` tools: `list_rooms`, `search_room(room)`,
  `inspect(object)`, `question(suspect, text)`, `lie_detector(statement_id)`
  (stub returning "unavailable" until M5), `accuse(suspect, motive?)`,
  `status()`. Good tool descriptions: they are prompts too.
- All suspects of the case active (one `suspect.py` agent each).
- `alibi play` in the terminal on top of the same tool functions.

**Acceptance criteria**
- [x] You can play a whole game in the terminal and win or lose.
- [x] You can play one game from an external MCP client (OpenCode or
      Claude Desktop). Document the config snippet in the README.
- [x] Unit tests: action counting, clue discovery rules, accusation scoring,
      invalid ids return friendly errors.
- [x] A game survives an app restart (state in Postgres).

**Learn:** MCP server design, tool descriptions as prompts, agent-friendly APIs.

---

### M4 — Evals (≈ 1 week)

**Goal:** measure the game automatically and block bad changes in CI.

**Tasks**
- `truth.py`: extract atomic claims from a suspect answer (structured
  output), label each **TRUE / FALSE / UNKNOWN** against the case. A FALSE
  claim is an **intentional lie** if it's about the speaker's secret or the
  murder (for the murderer), otherwise a **contradiction** (a bug).
  Store labels with each answer.
- `solver.py`: an automated detective (LangGraph) using only the MCP tools,
  with the same 20-action budget.
- `evals.py`: `alibi eval --games N` generates N cases, lets the solver play,
  computes metrics, writes `docs/results/eval-<date>.md`, and compares with
  `config/gates.yaml` (`--gate` exits non-zero on failure).
- Metrics (each with a docstring and a unit test on a fake transcript):

  | Metric | Meaning | Starting target |
  |---|---|---|
  | `case_valid_rate` | cases passing the validator | ≥ 80 % |
  | `solve_rate` | solver names the murderer | 40–85 % (too high = too easy) |
  | `contradiction_rate` | unintended false claims / all claims | < 8 % |
  | `early_confession_rate` | murderer confesses without 2 clues shown | < 10 % |
  | `cost_per_game` | $ per game | record it, then set a budget |

- Check the labeller: hand-label 30 claims, report agreement.
- GitHub Action on PRs touching `prompts/` or `suspect.py`: run a
  **5-game** eval with the cheapest model and fail on the gate. Post the
  summary table to the PR (e.g. `gh pr comment`).

**Acceptance criteria**
- [x] `alibi eval --games 20` runs unattended from your laptop.
- [x] Solver can't access ground truth (test on its inputs).
- [ ] Labeller vs your 30 hand labels: agreement reported (aim ≥ 80 %).
- [ ] A deliberately broken suspect prompt makes the CI gate fail. Keep that PR (closed) and link it in the README.
- [x] `docs/results/m4.md`: numbers, one failure you found, what you changed, before/after.

**Learn:** simulation-based evals, LLM-as-judge and checking the judge,
metric design, eval gates in CI.

---

### M5 — Lie Detector, Web UI and Homelab (≈ 1–1.5 weeks)

**Goal:** train a tiny model on your own game data, add a simple web UI,
and run it all on the N150.

**Lie detector**
- `alibi export-dataset`: from eval games, export `(statement, label)` with
  TRUE vs FALSE labels from `truth.py`. Split **by case** (no case in both
  train and test). Aim for ≥ 1,500 statements (run more eval games with the
  cheap model if needed).
- `ml/train.ipynb` (run on Colab/Kaggle): fine-tune DistilBERT with HF
  Transformers + PyTorch; compare with two baselines: always-TRUE and an
  LLM judge. Export to ONNX (`model.onnx` + tokenizer files).
- `detector.py`: ONNX Runtime inference on CPU. The `lie_detector` tool
  returns a fuzzy in-game hint ("the needle trembles").
- `ml/MODEL_CARD.md`: data, split, F1/precision/recall, 3 failure examples.

**Web UI**
- `web.py`: FastAPI + Jinja + HTMX. Pages: new game, room/evidence list,
  chat with a suspect, accuse, and **"What really happened"** (true timeline
  + each suspect's private "think" notes from the game).

**Homelab deploy**
- `compose.homelab.yml`: Postgres, app, Phoenix, with `mem_limit`s.
  App image based on `python:3.12-slim`, no `torch`.
- Public access via **Cloudflare Tunnel** (no open ports). Add a simple
  access code, per-IP rate limit, and a low daily spend cap.
- `docs/deploy-homelab.md`: step by step, including backup of the Postgres volume.

**Acceptance criteria**
- [x] Fine-tuned model beats always-TRUE on F1; comparison with the LLM
      judge reported honestly (quality, latency, cost per prediction).
- [ ] Detector inference < 100 ms per statement on the N150 (40 ms int8 on a laptop CPU).
- [ ] A friend can play a full game in the browser without instructions.
- [ ] Deployed stack uses < 2 GB RAM (`docker stats` screenshot in docs).
- [x] Spend cap tested: exceeding it shows a friendly "the detective is
      out of budget today" message.
- [ ] README: GIF, 3-command quickstart, architecture diagram, results table
      (M4 metrics + detector vs baselines), link to the demo.

**Learn:** dataset building and leakage, fine-tuning, ONNX serving on CPU,
shipping on constrained hardware, cost controls.

---

## 3. Stretch goals (only after M5)

- **Gossip**: suspects talk to each other when you're away and may align stories.
- **Model routing**: small model for minor characters, strong model for the
  murderer; compare cost and eval metrics.
- **More settings** (night train, jazz club) via YAML only.
- **Difficulty levels** tuned with `solve_rate`.
- **Nicer UI**, voice interrogation.

## 4. Interview notes (fill in as you go)

- Hardest problem: …
- A tradeoff between quality, latency and cost: …
- A bug the evals caught: …
- What I'd do next with a team: …

## 5. Progress

| Milestone | Done | Notes |
|---|---|---|
| M0 Skeleton | 🚧 | scaffolding done; hello + Phoenix trace verified |
| M1 Case File | 🚧 | code + tests done; 10/10 cases valid (see docs/results/m1.md); paper-solving pending |
| M2 One Suspect | ✅ | interrogate, pgvector memory, 4-node LangGraph; 10/10 injection-resistant (docs/results/m2.md) |
| M3 World + MCP | ✅ | play + MCP tools + Postgres state; think only for the murderer (docs/results/m3.md) |
| M4 Evals | 🚧 | code + gates + CI done; run shows solve_rate 1.0 (too easy); 20-game run & hand labels pending |
| M2 One Suspect | ⬜ | |
| M3 World + MCP | ⬜ | |
| M4 Evals | ⬜ | |
| M5 Detector + Web + Homelab | 🚧 | web UI + export + ONNX serving + deploy done; fine-tune pending (docs/results/m5.md) |