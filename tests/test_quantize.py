from __future__ import annotations

from pathlib import Path

import pytest

from alibi.quantize import quantize_model


def test_missing_model_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        quantize_model(tmp_path / "nope.onnx")


def test_quantizes_a_model_with_onnx_available(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")  # only present with the `ml` extra
    from onnx import TensorProto, helper

    weight = helper.make_tensor("W", TensorProto.FLOAT, [8, 8], [float(i) for i in range(64)])
    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, tmp_path / "model.onnx")

    before, after = quantize_model(tmp_path / "model.onnx")

    assert before > 0 and after > 0
    assert (tmp_path / "model.fp32.onnx").exists()  # fp32 kept for idempotency
