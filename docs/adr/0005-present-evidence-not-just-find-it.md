# 0005 — A suspect is shown evidence, not merely found out

## Context

The M4 evals showed the solver wins every game it finishes, and that prompt-only
difficulty changes did not help. The cause was in the confession mechanic:

- `Game` set each suspect's `clues_shown` to *every clue the detective had
  discovered* before every question.
- The murderer's prompt lets them confess once two pointing clues are "shown".
- So as soon as the detective had found two clues, any question to the murderer
  could trigger a confession, ending the game — no detective work required to
  connect a clue to a suspect.

Finding a clue and confronting someone with it are different acts.

## Decision

**"Shown" now means presented, not discovered.** `question()` takes an optional
`evidence=[clue_id, …]`; only those clues are recorded as shown to that suspect,
and only presented clues count toward the confession threshold. The MCP `question`
tool and the solver action gained the same `evidence` argument, and the terminal
has a `show <suspect_id> <clue_id…>` command. Evidence must have been found
first (friendly error otherwise).

## Alternatives

- **Add a separate `present_evidence` tool:** clearer, but grows the tool set;
  questioning-with-evidence is a natural single action.
- **Raise the threshold to 3 clues:** does not fix the conceptual bug (found ≠
  shown) and the solver finds clues trivially.
- **Rely on the prompt** ("only confess when the detective names the evidence"):
  already tried in spirit; prompts did not reliably change behaviour.

## Consequences

- The solver must now decide *which* clues implicate *which* suspect and present
  them — real detective work, and a lever the eval can measure.
- The mechanic matches the plan's wording ("confessing only when shown ≥2 clues
  that point to them").
- Existing games stored before this change still load (the new `presented` field
  defaults to empty).
