# 0007 — One game per visit, vendored web assets, and fair-play clue ids

## Context

Reviewing the browser game turned up three problems that were not about the
model at all:

- Game ids were the case id, so everyone who opened a case shared one game, and
  a finished case could never be replayed. Loading a game also saved a fresh
  state over the stored one.
- The HTMX `<script>` came from unpkg with an integrity hash that does not match
  htmx.org 2.0.3, so browsers refused to run it and no board action worked. The
  tests could not notice, because TestClient does not run JavaScript.
- Clue ids are shown to the detective (search results, MCP tools, the solver),
  and the generator is free to call a clue `clue_hobbs_glove`. A red herring
  aimed at an innocent was also labelled "nothing to do with you" in *that
  innocent's* prompt, so they shrugged off the very clue meant to rattle them.

## Decision

- The web UI starts a fresh game (`<case id>-<8 hex>`) on every "Start a game";
  the game URL is the save, and a cookie offers to continue the last one. The
  MCP server and CLI keep their explicit `--game-id`.
- htmx and the two fonts are vendored in `src/alibi/static` (OFL fonts, htmx is
  0BSD) and served by FastAPI. A test pins the htmx hash and forbids CDN scripts.
- A deterministic validator check, `clue_id_names_suspect`, rejects clue ids
  that contain a suspect's name or id words (titles like "lady" are ignored);
  the generator prompt states the rule. Suspects see "(this points at you)" for
  any clue aimed at them, red herring or not.

## Alternatives

- **Alias clue ids in the game layer** (`clue_1`…): hides names everywhere but
  touches the solver, transcripts and stored claims; the validator is smaller.
- **Fix the SRI hash and keep the CDN**: still breaks offline homelabs and
  strict CSPs, and depends on a third party at runtime.

## Consequences

- The two prompt changes affect suspect behaviour and generation, so they must
  pass the eval gate (`make eval`, or the PR workflow) before merging.
- One more validator check may cost an extra repair round now and then;
  `case_valid_rate` watches for that.
