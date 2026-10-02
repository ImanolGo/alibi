"""ONNX lie-detector inference on CPU.

The model is fine-tuned offline (see ``ml/``) and exported to ONNX together with
its tokenizer. It is loaded lazily, so the game runs fine with no model present
(the ``lie_detector`` tool then says it is not ready). No torch on the server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .config import get_settings


@dataclass(frozen=True)
class Prediction:
    label: str  # "FALSE" (a lie) or "TRUE"
    probability: float  # probability that the statement is FALSE


class LieDetector:
    """A tiny sequence classifier served with ONNX Runtime on CPU."""

    def __init__(self, session, tokenizer) -> None:
        self._session = session
        self._tokenizer = tokenizer

    @staticmethod
    def available(directory: Path | str) -> bool:
        path = Path(directory)
        return (path / "model.onnx").exists() and (path / "tokenizer.json").exists()

    @classmethod
    def load(cls, directory: Path | str) -> LieDetector:
        import onnxruntime
        from tokenizers import Tokenizer

        path = Path(directory)
        session = onnxruntime.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        return cls(session, tokenizer)

    @classmethod
    def from_settings(cls) -> LieDetector | None:
        """Load the configured model, or return None if it is not present."""
        directory = get_settings().detector_dir
        return cls.load(directory) if cls.available(directory) else None

    def predict(self, statement: str) -> Prediction:
        import numpy as np

        encoding = self._tokenizer.encode(statement)
        ids = np.array([encoding.ids], dtype="int64")
        mask = np.array([encoding.attention_mask], dtype="int64")

        inputs = {}
        for tensor in self._session.get_inputs():
            if "input_ids" in tensor.name:
                inputs[tensor.name] = ids
            elif "attention_mask" in tensor.name:
                inputs[tensor.name] = mask
            else:
                inputs[tensor.name] = np.zeros_like(ids)

        logits = self._session.run(None, inputs)[0][0]
        # Softmax; class 1 is FALSE (the fine-tuning label).
        shift = max(float(value) for value in logits)
        exps = [math.exp(float(value) - shift) for value in logits]
        total = sum(exps)
        probability_false = exps[1] / total if len(exps) > 1 else exps[0] / total
        return Prediction(
            label="FALSE" if probability_false >= 0.5 else "TRUE",
            probability=probability_false,
        )
