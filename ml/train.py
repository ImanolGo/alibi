"""Fine-tune a tiny lie detector on the exported claims.

Run this on a laptop with a GPU, Colab or Kaggle — never on the server.

    pip install torch transformers
    uv run alibi export-dataset                 # writes ml/data/*.jsonl
    python ml/train.py --data ml/data --out ml/model

It compares against two baselines (always-TRUE and a majority-class guess) and
exports ``model.onnx`` + ``tokenizer.json`` for ONNX Runtime on the server, so
``torch`` never ships to the homelab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_NAME = "distilbert-base-uncased"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}; run `alibi export-dataset` first")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def f1_of_lies(true_labels: list[int], predicted: list[int]) -> float:
    """F1 for the positive class (1 = a lie)."""
    tp = sum(1 for t, p in zip(true_labels, predicted, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, predicted, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, predicted, strict=True) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(model, loader, torch, device) -> tuple[list[int], list[int]]:
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch, labels in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = labels.to(device)
            logits = model(**batch).logits
            predictions.extend(int(p) for p in logits.argmax(dim=-1))
            truths.extend(int(t) for t in labels)
    return truths, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ml/data")
    parser.add_argument("--out", default="ml/model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train_rows = read_jsonl(Path(args.data) / "train.jsonl")
    val_rows = read_jsonl(Path(args.data) / "val.jsonl")
    test_path = Path(args.data) / "test.jsonl"
    test_rows = read_jsonl(test_path) if test_path.exists() else val_rows
    print(f"{len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test statements")

    class Claims(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            encoded = tokenizer(
                row["statement"],
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {key: value[0] for key, value in encoded.items()}, torch.tensor(row["label"])

    train_loader = DataLoader(Claims(train_rows), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Claims(val_rows), batch_size=args.batch_size)
    test_loader = DataLoader(Claims(test_rows), batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for batch, labels in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = model(**batch, labels=labels).loss
            loss.backward()
            optimizer.step()
        truths, predictions = evaluate(model, val_loader, torch, device)
        print(f"epoch {epoch + 1}: val F1(lie) = {f1_of_lies(truths, predictions):.3f}")

    # Baselines, for the model card.
    baseline_truths = [row["label"] for row in test_rows]
    always_true = [0] * len(baseline_truths)
    majority_label = max((0, 1), key=lambda label: baseline_truths.count(label))
    majority_all = [majority_label] * len(baseline_truths)
    print(f"always-TRUE baseline F1(lie) = {f1_of_lies(baseline_truths, always_true):.3f}")
    print(f"majority baseline F1(lie)    = {f1_of_lies(baseline_truths, majority_all):.3f}")

    truths, predictions = evaluate(model, test_loader, torch, device)
    print(f"fine-tuned TEST F1(lie)      = {f1_of_lies(truths, predictions):.3f}")

    # Export to ONNX + tokenizer for CPU inference on the server.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    model.eval()
    dummy = tokenizer(
        "example statement",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    tokenizer.save_pretrained(out)
    print(f"wrote {out}/model.onnx and tokenizer files")


if __name__ == "__main__":
    main()
