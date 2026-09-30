# 0002 — Suspects: private reasoning, pgvector memory and information hiding

## Context

A suspect must stay in character, remember the interrogation, and lie — but
only about its own secret (or, for the murderer, the crime). If a suspect's
prompt contained the whole case it would know too much and could not plausibly
be ignorant; if it had no memory it would contradict itself within a few turns.

## Decision

- **Information hiding is a code guarantee, not a prompt wish.** `knowledge_for`
  returns only the events a person took part in or witnessed plus their own
  secrets. `SuspectAgent.system_prompt()` renders *only* that, the persona and
  (for the murderer alone) the fact that they did it. A unit test asserts the
  innocent's prompt contains no other secret and never the word "murderer".
- **LangGraph with four nodes:** `retrieve` (top-k from pgvector) → `think`
  (private structured `Thought`) → `speak` (in-character answer) → `remember`
  (store question + answer). Each node gets a trace span under `suspect.turn`.
- **Memory:** `memories(game_id, suspect_id, content, embedding vector(1536))`
  in Postgres/pgvector; `remember()`/`recall()` embed through `llm.embed`.
- **Confession:** the murderer may confess only once the detective has shown at
  least two non-red-herring clues that point at them.

## Alternatives

- **One LLM call per turn** (no private think): simpler, but the model tends to
  leak secrets or contradict itself; hidden reasoning measurably steadies the
  deception.
- **Full transcript in the prompt**: does not scale and makes hiding fragile;
  instead we retrieve relevant memories semantically.
- **Put secrets in the prompt as "do not reveal"**: works until the model is
  pushed; keeping them out of the prompt entirely is stronger.

## Consequences

- The talkative "think" output is available behind `--debug` and is exactly the
  kind of private note the "what really happened" ending will show later.
- Retrieval costs one embedding per turn; cheap and enough for a single suspect.
- Memory is per `(game_id, suspect_id)`, so several games/suspects coexist.
