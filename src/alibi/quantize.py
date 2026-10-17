"""Dynamic int8 quantization for the ONNX lie detector.

An offline step (run on a laptop, not the server): it shrinks ``model.onnx``
from ~268 MB fp32 to ~67 MB int8 and speeds up CPU inference. The fp32 file is
kept as ``model.fp32.onnx`` so the step is idempotent. ``onnxruntime.quantization``
imports the ``onnx`` package, which lives in the ``ml`` group, not the server image.
"""

from __future__ import annotations

from pathlib import Path


class QuantizeUnavailableError(RuntimeError):
    """Raised when the ``ml`` dependency group (which provides ``onnx``) is missing."""


def quantize_model(model_path: Path | str, out_dir: Path | str | None = None) -> tuple[int, int]:
    """Quantize ``model_path`` to int8 in ``out_dir``; return (before, after) bytes."""
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(source)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:  # onnx is not installed in the default environment
        raise QuantizeUnavailableError(
            "quantizing needs the `onnx` package: run `uv sync --group ml` first"
        ) from exc

    destination = Path(out_dir) if out_dir else source.parent
    destination.mkdir(parents=True, exist_ok=True)

    fp32 = destination / "model.fp32.onnx"
    if not fp32.exists():
        fp32.write_bytes(source.read_bytes())

    target = destination / "model.onnx"
    quantize_dynamic(str(fp32), str(target), weight_type=QuantType.QInt8)
    return fp32.stat().st_size, target.stat().st_size
