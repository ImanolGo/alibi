# M4 — Labeller agreement (hand labels)

Check: does `truth.py`'s judge agree with a careful human reading of the case?

- Case: `cases/blackwood_manor_1926.json` (murderer `dr_vance`; ground truth printed
  from the case file — timeline, clues, secrets).
- Sample: 30 atomic claims the judge extracted from suspects' answers to two
  questions each.
- Method: the claims were frozen to a file with the judge's verdicts hidden;
  they were then labelled by hand (TRUE / FALSE / UNKNOWN) against the case's
  ground truth and compared.
- The judge is an LLM, so this is a second, independent LLM reading — not a
  human one; it still catches systematic bias and gross errors.

## Agreement: 27 / 30 = **90%** (target ≥ 80%) ✅

| # | Judge | Hand | Claim |
|---|---|---|---|
| 0 | UNKNOWN | UNKNOWN | I have already told you my whereabouts. |
| 1 | **FALSE** | **TRUE** | I was at the whist table in the drawing room from a quarter to ten. |
| 2 | TRUE | TRUE | I was at the whist table in the drawing room until the rubber ended at half past. |
| 3 | TRUE | TRUE | I was with Hugh, Miss Thorne and Dr Vance at the whist table. |
| 4 | FALSE | FALSE | No one rose from the whist table. |
| 5 | UNKNOWN | UNKNOWN | The hour of death is a question for the doctor who attended him last. |
| 6 | UNKNOWN | UNKNOWN | I am a woman holding a hand of cards. |
| 7 | UNKNOWN | UNKNOWN | I am not accustomed to repeating myself. |
| 8 | UNKNOWN | UNKNOWN | I am happy to repeat myself if it saves you the trouble… |
| 9 | UNKNOWN | UNKNOWN | Unusual depends on your standards, Detective. |
| 10 | UNKNOWN | UNKNOWN | On a snowed-in evening the unusual is rather the order of the day. |
| 11 | TRUE | TRUE | Edmund sulked through dinner over the accounts. |
| 12 | UNKNOWN | UNKNOWN | The servants will confirm that this is his usual manner… |
| 13 | **TRUE** | **UNKNOWN** | It is his usual manner of addressing the company. |
| 14 | TRUE | TRUE | Dr Vance kept the whist table waiting until five past ten. |
| 15 | TRUE | TRUE | Dr Vance had looked in on his patient. |
| 16 | FALSE | FALSE | Dr Vance heard the bolt shot behind him. |
| 17 | UNKNOWN | UNKNOWN | Beyond that, nothing I should care to call remarkable. |
| 18 | TRUE | TRUE | Hugh proposed a nightcap before the tray had come down. |
| 19 | TRUE | TRUE | I was at the whist table from the cut at ten until the rubber ended. |
| 20 | TRUE | TRUE | Lady Constance was at the whist table with me. |
| 21 | TRUE | TRUE | Miss Thorne was at the whist table with me. |
| 22 | TRUE | TRUE | Dr Vance was at the whist table with me. |
| 23 | UNKNOWN | UNKNOWN | I could not tell you when the poor devil actually died. |
| 24 | TRUE | TRUE | Vance says the heart gave out. |
| 25 | UNKNOWN | UNKNOWN | I was losing fourpence a point at the time. |
| 26 | **TRUE** | **UNKNOWN** | The bolt is unusual. |
| 27 | TRUE | TRUE | Vance made rather a point of telling us he had heard the bolt shot… |
| 28 | TRUE | TRUE | Dr Vance described the patient as being in excellent spirits. |
| 29 | UNKNOWN | UNKNOWN | It is curious for a man to remember that about a patient… |

## The three disagreements

- **#1 (judge FALSE, hand TRUE).** The judge was wrong: the timeline has Lady
  Constance at the whist table from the cut at 21:55, so her claim is true. It
  likely over-indexed on Dr Vance being *absent* at 22:00 — a different person.
- **#13 (judge TRUE, hand UNKNOWN).** The judge was right and I was wrong: the
  timeline says Edmund's silence was "his way of speaking to everybody". The
  judge used a detail I had skimmed. Good sign.
- **#26 (judge TRUE, hand UNKNOWN).** Debatable: "the bolt is unusual" is a
  judgement, but the case does stress the unshot bolt as the anomaly. Either
  label is defensible.

## Takeaway

The judge is good at the things that matter for the metrics: it labelled the two
real falsehoods (#4 nobody rose; #16 the bolt heard) as FALSE, and left opinions
as UNKNOWN. The failures are on under-specified or evaluative statements — worth
remembering when reading `contradiction_rate`, since a borderline claim can flip
a contradiction.
