# 0006 — Enforce ambiguity: innocents must be equally implicated

## Context

M4 showed the solver won every completed game no matter how the confession
mechanic or prompts changed. Whatever the wording, the generated clues still
narrowed to a single suspect: the culprit was the only person a clue pointed at,
so the deduction was trivial.

## Decision

Add a seventh deterministic validator check, `insufficient_spread`: at least two
red-herring clues must point at **two different innocents** (via `points_to`).
Together with the existing "≥2 real clues point at the murderer", this forces
the culprit to be one of several implicated people. The generator prompt states
the same requirement, and the repair loop enforces it.

## Alternatives

- **Prompt-only** ("make it ambiguous"): already tried in M4; the model happily
  produces a single-suspect case.
- **Weaken the solver** (fewer actions, less information): does not fix the
  game a human player would also find too easy.
- **Accept the high solve rate and raise the gate ceiling**: honest but gives up
  on the difficulty the design wanted (40–85%).

## Consequences

- Cases now have at least two plausible red herrings aimed at innocents, so the
  detective (human or solver) must weigh evidence rather than tally nametags.
- More constraints mean the generator may fail validation more often; the
  bounded repair loop and `case_valid_rate` gate watch for that.
- This is a deliberate departure from the plan's "keep it to these six checks";
  recorded here so the deviation is explicit.
