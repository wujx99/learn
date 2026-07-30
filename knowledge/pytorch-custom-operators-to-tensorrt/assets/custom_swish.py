"""A training-capable custom operator implemented through torch.library."""

from __future__ import annotations

import torch
from torch import Tensor


@torch.library.custom_op("learn_ops::swish_bias", mutates_args=())
def swish_bias(x: Tensor, bias: Tensor, beta: float) -> Tensor:
    """Compute x * sigmoid(beta * x) + bias; bias is a scalar Tensor."""
    if bias.numel() != 1:
        raise RuntimeError("bias must contain exactly one element")
    return x * torch.sigmoid(beta * x) + bias


@swish_bias.register_fake
def _(x: Tensor, bias: Tensor, beta: float) -> Tensor:
    torch._check(bias.numel() == 1)
    torch._check(x.device == bias.device)
    torch._check(x.dtype == bias.dtype)
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output) -> None:
    x, bias, beta = inputs
    ctx.save_for_backward(x, bias)
    ctx.beta = beta


def _backward(ctx, grad_output: Tensor):
    x, bias = ctx.saved_tensors
    beta = ctx.beta
    sigmoid = torch.sigmoid(beta * x)
    grad_x = grad_output * (sigmoid + beta * x * sigmoid * (1 - sigmoid))
    grad_bias = grad_output.sum().reshape_as(bias)
    return grad_x, grad_bias, None


swish_bias.register_autograd(_backward, setup_context=_setup_context)
