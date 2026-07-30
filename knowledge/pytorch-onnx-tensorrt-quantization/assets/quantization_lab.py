"""Small CPU lab for affine quantization, calibration, and per-channel weights."""

from __future__ import annotations

import torch


def symmetric_qparams(x: torch.Tensor, qmax: int = 127):
    scale = x.abs().amax().clamp_min(torch.finfo(torch.float32).eps) / qmax
    return scale


def fake_quant_symmetric(x: torch.Tensor, scale: torch.Tensor, qmax: int = 127):
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def per_channel_weight_fake_quant(weight: torch.Tensor):
    # Linear weight layout is [out_features, in_features].
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).eps
    ) / 127
    return fake_quant_symmetric(weight, scale)


def report(name: str, actual: torch.Tensor, expected: torch.Tensor):
    error = actual - expected
    print(
        f"{name:28s} mse={error.square().mean():.3e} "
        f"max_abs={error.abs().max():.3e}"
    )


def main() -> None:
    torch.manual_seed(42)

    # Calibration data is mostly N(0, 1); evaluation includes an outlier.
    calibration = torch.randn(256, 32)
    evaluation = torch.randn(64, 32)
    evaluation[0, 0] = 12.0

    calibration_scale = symmetric_qparams(calibration)
    oracle_scale = symmetric_qparams(evaluation)
    report(
        "activation/calibration scale",
        fake_quant_symmetric(evaluation, calibration_scale),
        evaluation,
    )
    report(
        "activation/oracle scale",
        fake_quant_symmetric(evaluation, oracle_scale),
        evaluation,
    )
    saturation = (evaluation.abs() > calibration_scale * 127).float().mean()
    print(f"calibration saturation ratio={saturation:.3%}")

    weight = torch.randn(16, 32)
    weight[0] *= 20  # one output channel has a very different range
    per_tensor = fake_quant_symmetric(weight, symmetric_qparams(weight))
    per_channel = per_channel_weight_fake_quant(weight)
    report("weight/per-tensor", per_tensor, weight)
    report("weight/per-output-channel", per_channel, weight)

    x = torch.randn(8, 32)
    reference = x @ weight.t()
    report("linear/per-tensor weight", x @ per_tensor.t(), reference)
    report("linear/per-channel weight", x @ per_channel.t(), reference)


if __name__ == "__main__":
    main()
