# 0003 — The game as MCP tools, and where to spend the `think` call

## Context

M3 needs a complete game the player drives through a fixed action set, exposed
both in the terminal and to any MCP client. Each suspect turn costs two chat
calls (think + speak) plus two embeddings, so a four-suspect interrogation is
slow and adds up.

## Decision

- **One `Game` object owns the rules and is the single source of truth.** The
  terminal (`alibi play`) and the MCP server (`alibi mcp`) are thin adapters
  over the same methods, so they cannot drift apart.
- **Costly actions run through a small LangGraph pipeline**
  (`validate → apply → update`). Validation returns friendly errors and, when
  it fails, the action is free; `update` decrements and persists.
- **State lives in Postgres** (`games` row = case JSON + state JSON), so a game
  survives a restart and the same game can be resumed from either front end.
- **The private `think` step is kept only for the murderer.** Innocents answer
  straight from the persona; the murderer, whose consistency matters, still pays
  for the structured call. Configurable via `SuspectAgent(reason=...)` and
  re-checked against the injection suite.
- **The lie detector is a free stub** until M5 — charging the player for a tool
  that does nothing would be poor design.

## Alternatives

- **Two separate implementations** for terminal and MCP: guaranteed to diverge.
- **Think for everyone:** ~2× the latency for the three suspects who least need
  it, per the M2 results.
- **No think at all:** fastest, but the murderer starts to slip (exactly what
  the M4 `early_confession_rate` gate watches).
- **In-memory only game state:** fails the "survives a restart" criterion.

## Consequences

- Adding a new action means one method in `game.py` plus one thin MCP tool.
- Tool docstrings are prompts: they are how an LLM detective knows what it can
  afford to do, so they mention the action cost.
- Innocents lose their hidden reasoning; the M2 info-hiding and injection tests
  are re-run with `reason=False` to make sure nothing regressed.
