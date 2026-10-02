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

## Results (validation)

| Model | F1 (lie) | Precision | Recall |
|---|---|---|---|
| always-TRUE baseline | TODO | TODO | TODO |
| majority baseline | TODO | TODO | TODO |
| LLM judge (`truth.py`) | TODO | TODO | TODO |
| **fine-tuned (this model)** | **TODO** | TODO | TODO |

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
