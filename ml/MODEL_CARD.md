# Model card — Alibi lie detector

> Fill this in after training on Colab/Kaggle. Numbers below are the template.

## What it is

A small sequence classifier that reads one statement and decides whether it is
**TRUE** or **FALSE** with respect to the case. In the game it powers the
`lie_detector` tool, which returns a fuzzy hint ("the needle trembles").

## Training data

- Source: `alibi export-dataset`, from claims labelled by `truth.py` during
  `alibi eval` games. Each eval game is one case.
- Split **by case** (no case appears in two splits) to avoid leakage.
- Statements: **TODO** (aim ≥ 1,500). Class balance (lie/truth): **TODO**.
- Base model: `distilbert-base-uncased`, max length 128.
- Fine-tuning: `ml/train.ipynb` (Colab/Kaggle, one click) or `ml/train.py`.

## Results (test split)

Baselines come from `alibi baselines --data ml/data/test.jsonl`; the fine-tuned
numbers come from `ml/train.ipynb`. The LLM judge sees the same input as the
model (one statement, no case context), so it is a deliberately hard baseline.

| Model | F1 (lie) | Latency | Cost / prediction |
|---|---|---|---|
| always-TRUE baseline | 0.000 | — | — |
| majority baseline | 0.000 | — | — |
| LLM judge (`alibi baselines`) | 0.182 | 7.7 s | $0.00022 |
| **fine-tuned ONNX (this model)** | **TODO** | aim < 0.1 s | ~$0 |

> The LLM judge number is a 30-statement sample; the always-TRUE/majority F1 of
> 0 means the test split is balanced enough that guessing one class scores
> nothing on the lie class. Every model must beat chance on the statements
> alone — which is the point of a fine-tuned specialist.

## Serving

- Exported to `model.onnx` + `tokenizer.json`, run with ONNX Runtime on CPU
  (no `torch` on the server). Target < 100 ms per statement on the N150.
- `detector.py` loads it from `ALIBI_DETECTOR_DIR` (default `ml/model`).

## Failure examples

Three cases where the model gets it wrong (write these in):

1. TODO
2. TODO
3. TODO

## Limitations

- Trained on synthetic cosy-mystery statements from one setting; it will not
  generalise to other domains.
- It judges claims in isolation (no conversation context).
- Class imbalance and label noise from the LLM judge are the main risks.
