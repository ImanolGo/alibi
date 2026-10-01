# 0004 — Truth labelling and eval metrics

## Context

To know whether a change to the suspect prompts or agent made things better or
worse, we need numbers: is the solver too good (case too easy), do innocents
contradict themselves, does the murderer confess too early? "Lies" also need to
be distinguished from bugs: a murderer denying the crime is *expected*; an
innocent mis-stating the timeline is a defect.

## Decision

- **A judge, not the solver, sees the ground truth.** `truth.py` asks an LLM
  (role `judge`) to split an answer into atomic claims and label each
  TRUE / FALSE / UNKNOWN, to say whether it concerns the speaker's secret or the
  murder, and whether the speaker confessed. Ground truth is passed to the judge
  only — never to the solver.
- **Classification is deterministic.** `claim_kind()` (pure, unit-tested) maps a
  FALSE claim to `intentional_lie` when it is about the speaker's secret or,
  for the murderer, the murder; anything else FALSE is a `contradiction`.
- **Metrics are pure functions of per-game records** (`evals.py`), so they are
  tested on fake transcripts with no model. `alibi eval --games N` runs the
  generator, the solver and the judge, writes `docs/results/eval-<date>.md`, and
  `--gate` exits non-zero against `config/gates.yaml`.
- **A separate CI workflow** runs a 5-game gate on PRs touching `prompts/` or
  `suspect.py` and comments the summary on the PR.

## Alternatives

- **Keyword heuristics for lies:** cheap but brittle; misses paraphrase.
- **Judge labels both verdict and kind:** two LLM judgements where one
  deterministic rule suffices; harder to test.
- **Let the solver self-report:** it cannot — it never sees the truth.

## Consequences

- `contradiction_rate` is the metric that catches prompt regressions; a
  deliberately broken suspect prompt should push it over the gate.
- Judge quality matters, so its agreement with ~30 hand-labelled claims is
  measured and recorded (`docs/results/m4_labels.md`).
- Evaluating costs money; the runner stops on `BudgetExceededError` and reports
  the games it managed to play.
