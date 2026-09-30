# M4 — Labeller agreement (hand labels)

Hand-label 30 claims taken from real suspect answers, then compare with what
`truth.py` (the judge) produced. Fill the table in.

## How to produce claims to label

```bash
uv run alibi play --case cases/<id>.json --new        # ask suspects a few things
# or from a saved eval:
uv run python -c "from alibi.db import ...; ..."       # read the claims table
```

Every labelled claim is stored in Postgres (`claims` table) with its `game_id`,
`suspect_id`, `answer`, `claim`, `verdict` and `kind`.

## Agreement

| # | claim | judge verdict | your verdict | agree? |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| … | | | | |
| 30 | | | | |

- Agreement: **TODO / 30** (aim ≥ 80%)
- Notes on disagreements (systematic biases worth remembering): **TODO**
