"""Validate ONNX structure and compare ORT outputs against golden PyTorch outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = args.artifacts / "demo_dynamic.onnx"
    onnx.checker.check_model(onnx.load(onnx_path))
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    cases = torch.load(args.artifacts / "golden_io.pt", weights_only=True)

    for batch_size, case in cases.items():
        actual = session.run(
            ["logits"], {"images": case["images"].numpy()}
        )[0]
        expected = case["logits"].numpy()
        np.testing.assert_allclose(
            actual, expected, rtol=args.rtol, atol=args.atol
        )
        max_abs = float(np.max(np.abs(actual - expected)))
        print(f"batch={batch_size:2d} shape={actual.shape} max_abs={max_abs:.3e} PASS")


if __name__ == "__main__":
    main()
