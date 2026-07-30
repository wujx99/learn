"""Translate the custom PyTorch op into a standard ONNX subgraph and verify it."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxscript import FLOAT
from onnxscript import opset18 as op

from custom_swish import swish_bias


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.125))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish_bias(x, self.bias, 1.25)


def onnx_swish_bias(x: FLOAT, bias: FLOAT, beta: float) -> FLOAT:
    beta_tensor = op.CastLike(beta, x)
    return op.Add(op.Mul(x, op.Sigmoid(op.Mul(beta_tensor, x))), bias)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("swish_bias.onnx"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = Model().eval()
    example = torch.randn(8, 16)
    batch = torch.export.Dim("batch", min=1, max=16)
    program = torch.onnx.export(
        model,
        (example,),
        input_names=["x"],
        output_names=["y"],
        opset_version=18,
        dynamo=True,
        dynamic_shapes={"x": {0: batch}},
        custom_translation_table={
            torch.ops.learn_ops.swish_bias.default: onnx_swish_bias,
        },
    )
    program.save(args.output)
    onnx.checker.check_model(onnx.load(args.output))

    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    for batch_size in (1, 8, 16):
        x = torch.randn(batch_size, 16)
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = session.run(["y"], {"x": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        max_abs = float(np.max(np.abs(actual - expected)))
        print(f"batch={batch_size:2d} max_abs={max_abs:.3e} PASS")


if __name__ == "__main__":
    main()
