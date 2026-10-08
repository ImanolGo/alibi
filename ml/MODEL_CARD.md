# Model card — Alibi lie detector

## What it is

A small sequence classifier that reads one statement and decides whether it is
**TRUE** or **FALSE** with respect to the case. In the game it powers the
`lie_detector` tool, which returns a fuzzy hint ("the needle trembles").

## Training data

- Source: `alibi export-dataset`, from claims labelled by `truth.py` during
  `alibi eval` games. Each eval game is one case.
- Split **by case** (no case appears in two splits) to avoid leakage.
- Statements: **1,271** (train 985 / val 130 / test 156). Test split:
  25 lies, 131 truths — **imbalanced ~1:5**.
- Base model: `distilbert-base-uncased`, max length 128, 3 epochs, lr 2e-5.
- Fine-tuning: `ml/train.py` (also a one-click `ml/train.ipynb` for Colab).
  Trained on a GTX 1650 in ~1 minute.

## Results (test split, n=156)

Baselines from `alibi baselines --data ml/data/test.jsonl`; the fine-tuned row
from `ml/train.py`. The LLM judge sees the same input as the model (one
statement, no case context), so it is a deliberately hard baseline.

| Model | F1 (lie) | Latency | Cost / prediction |
|---|---|---|---|
| always-TRUE baseline | 0.000 | — | — |
| majority baseline | 0.000 | — | — |
| LLM judge (`alibi baselines`) | 0.182 | 7.7 s | $0.00022 |
| **fine-tuned ONNX (this model)** | **0.333** | **63 ms** | **~$0** |

The fine-tuned model **beats always-TRUE** (0.333 vs 0.000) and the LLM judge
(0.333 vs 0.182), and is ~120× faster and effectively free. The F1 is modest
because the data is small and imbalanced; more eval games (≥ 1,500–5,000
statements) and class weighting are the obvious next steps. latency measured on
a laptop CPU (12 cores); the N150 will be slower.

## Serving

- Exported to `model.onnx` (268 MB, fp32) + `tokenizer.json`, run with ONNX
  Runtime on CPU (no `torch` on the server).
- `detector.py` loads it from `ALIBI_DETECTOR_DIR` (default `ml/model`).
- **Follow-up:** 268 MB is heavy for the 400 MB app budget; int8 dynamic
  quantization (`onnxruntime.quantization`) should cut it to ~67 MB.

## Failure examples

Real test-set errors (the model does not know the case, only the sentence):

1. **False positive** (called a lie, was TRUE): *"Every altered figure in the
   ledger is mine."* — reads like a confession, but it is in fact true.
2. **False positive**: *"I was never once late for anything."* — an emphatic
   denial pattern that is usually a lie in the training data.
3. **False negative** (missed a lie): *"I had left the library some quarter of
   an hour before ten."* — a matter-of-fact alibi, no hedging to latch onto.

## Limitations

- Trained on synthetic cosy-mystery statements from one setting; it will not
  generalise to other domains.
- It judges claims in isolation (no conversation context), so a true-and-damning
  sentence and a false-and-innocent one look alike. It leans on style.
- Class imbalance (~1:5) and label noise from the LLM judge are the main risks.
