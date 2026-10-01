# 0001 — Generate cases with a structured call, gate them with a validator

## Context

The case generator must invent a whole mystery (people, secrets, a timeline,
clues) that is *internally consistent*: the murderer is at the scene at the
time of death, nobody is in two places at once, and enough clues point to the
culprit. A raw free-text generation would be creative but often broken, and
asking the model to "double-check itself" is unreliable.

## Decision

Split the job in two:

1. **Creativity (LLM):** one `llm.structured(...)` call returns a `Case`
   (Pydantic) — the model fills the schema from `config/setting.yaml`.
2. **Correctness (code):** `validator.validate(case)` runs six deterministic
   checks and returns a list of `Violation`s.

If there are violations, we feed the case *and the exact violations* back to the
model in a repair prompt, at most 2 times. If it still fails, we return the case
with its violations rather than looping forever or silently accepting it.

## Alternatives

- **Pure prompt engineering** ("make sure the times are consistent"): no
  guarantee, untestable.
- **Constraint solving / programmatic generation:** fully correct but loses the
  flavour and variety that make each mystery worth reading.
- **Unbounded repair loop:** can burn budget on a hopeless case.

## Consequences

- Validation is fully unit-testable with hand-made cases, no LLM needed.
- The repair loop is bounded and observable (each iteration is logged).
- The model may still produce a case we can't fix; `alibi case` exits non-zero
  and saves the JSON so we can inspect the failure.
