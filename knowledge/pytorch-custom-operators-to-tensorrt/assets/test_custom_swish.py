"""Run correctness, dispatcher, FakeTensor, autograd, and compile checks."""

from __future__ import annotations

import torch

from custom_swish import swish_bias


def reference(x: torch.Tensor, bias: torch.Tensor, beta: float) -> torch.Tensor:
    return x * torch.sigmoid(beta * x) + bias


def main() -> None:
    torch.manual_seed(42)
    x = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    bias = torch.randn((), dtype=torch.double, requires_grad=True)
    beta = 1.25

    torch.testing.assert_close(swish_bias(x, bias, beta), reference(x, bias, beta))
    print("forward: PASS")

    assert torch.autograd.gradcheck(
        lambda a, b: swish_bias(a, b, beta), (x, bias), eps=1e-6, atol=1e-5
    )
    print("gradcheck: PASS")

    torch.library.opcheck(swish_bias, (x, bias, beta))
    print("opcheck: PASS")

    compiled = torch.compile(lambda a, b: swish_bias(a, b, beta), fullgraph=True)
    torch.testing.assert_close(compiled(x, bias), reference(x, bias, beta))
    print("torch.compile: PASS")

    if torch.cuda.is_available():
        x_cuda = x.detach().float().cuda().requires_grad_()
        bias_cuda = bias.detach().float().cuda().requires_grad_()
        swish_bias(x_cuda, bias_cuda, beta).sum().backward()
        print("CUDA dispatch/autograd: PASS")


if __name__ == "__main__":
    main()
