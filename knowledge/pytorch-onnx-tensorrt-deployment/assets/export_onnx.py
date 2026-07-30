"""Create a checkpoint, golden cases, and a dynamic-batch ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from demo_model import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "export_report").mkdir(parents=True, exist_ok=True)

    model = make_model().eval()
    checkpoint_path = args.output_dir / "demo_checkpoint.pth"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    restored = make_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored.eval()

    generator = torch.Generator().manual_seed(2026)
    cases = {}
    with torch.inference_mode():
        for batch_size in (1, 8, 16):
            images = torch.randn(batch_size, 3, 32, 32, generator=generator)
            cases[batch_size] = {
                "images": images,
                "logits": restored(images),
            }
    torch.save(cases, args.output_dir / "golden_io.pt")

    batch = torch.export.Dim("batch", min=1, max=16)
    onnx_program = torch.onnx.export(
        restored,
        (cases[8]["images"],),
        input_names=["images"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=True,
        dynamic_shapes={"images": {0: batch}},
        report=True,
        artifacts_dir=str(args.output_dir / "export_report"),
    )
    onnx_path = args.output_dir / "demo_dynamic.onnx"
    onnx_program.save(onnx_path)
    print(f"checkpoint: {checkpoint_path}")
    print(f"golden cases: {args.output_dir / 'golden_io.pt'}")
    print(f"ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
