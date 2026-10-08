# Deploying Alibi on the homelab (N150, 4 GB RAM, no GPU)

The stack is three containers: **Postgres + pgvector** (memory and game state),
**Phoenix** (tracing) and **the app** (the game exposed as MCP tools over
streamable HTTP). No local model runs on the box — all LLMs and embeddings go to
hosted APIs through LiteLLM. Total budget is under ~2 GB.

> The browser UI is milestone M5 and is not built yet. Today the deployed
> service is the MCP endpoint; a local MCP client (or the OpenCode/Claude
> Desktop MCP config) can play it over HTTP.

## 1. Prerequisites

- Docker + Docker Compose on the VM.
- An OpenRouter API key (chat **and** embeddings).
- Optional: a Cloudflare Tunnel for public access without opening ports.

## 2. Configure

```bash
git clone <your repo> alibi && cd alibi
cp .env.example .env
# edit .env: set OPENROUTER_API_KEY, keep ALIBI_DAILY_BUDGET_USD low (e.g. 1.0)
```

`compose.homelab.yml` reads `OPENROUTER_API_KEY` from the environment via `.env`.

## 3. Generate a case (once)

The app serves the newest case in its `cases` volume:

```bash
docker compose -f compose.homelab.yml run --rm app alibi case --seed 1
```

## 4. Start

```bash
make deploy          # docker compose -f compose.homelab.yml up -d --build
docker stats --no-stream   # confirm each service stays within its mem_limit
```

- MCP endpoint: `http://<vm>:8089` (streamable HTTP) — for OpenCode / Claude
  Desktop.
- Web UI: `http://<vm>:8080` — play in a browser.
- Phoenix UI: `http://<vm>:6006` (keep this local; it shows your traces).

## 5. Point a client at it

OpenCode / Claude Desktop MCP config (HTTP):

```json
{
  "mcpServers": {
    "alibi": { "type": "http", "url": "http://<vm>:8089/mcp" }
  }
}
```

Play in the terminal against the same Postgres with:

```bash
uv run alibi play --case cases/<id>.json
```

## 6. Public access (Cloudflare Tunnel)

Install `cloudflared`, create a tunnel, and route a hostname to
`http://localhost:8080` — no inbound ports opened. Add an access policy (a
simple access code or Cloudflare Access) in front of it. Keep the daily spend
cap low: the app is a public toy, not a free LLM proxy.

## 7. Cost and safety

- `ALIBI_DAILY_BUDGET_USD` is a hard cap; calls that would exceed it raise
  `BudgetExceededError` and the player sees a friendly message.
- Suspect memory and game state live in Postgres, so the game resumes after a
  restart.

## 8. Backup

```bash
docker compose -f compose.homelab.yml exec postgres \
  pg_dump -U alibi alibi | gzip > alibi-$(date +%F).sql.gz
```

Restore with `gunzip -c alibi-<date>.sql.gz | docker compose -f compose.homelab.yml exec -T postgres psql -U alibi alibi`.

## 9. Update

```bash
git pull
make deploy   # rebuilds the app image; volumes are preserved
```

## Memory budget

| Service | `mem_limit` |
|---|---|
| Postgres + pgvector | 350 MB |
| Phoenix | 450 MB |
| App (FastAPI + ONNX detector later) | 400 MB |
| OS + Docker | ~400 MB |
| **Total** | **< 2 GB** |
