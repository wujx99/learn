"""Minimal autograd.Function version used to study forward/backward mechanics."""

from __future__ import annotations

import torch


class SwishBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, bias: torch.Tensor, beta: float):
        ctx.save_for_backward(x, bias)
        ctx.beta = beta
        return x * torch.sigmoid(beta * x) + bias

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, bias = ctx.saved_tensors
        sigmoid = torch.sigmoid(ctx.beta * x)
        grad_x = grad_output * (
            sigmoid + ctx.beta * x * sigmoid * (1 - sigmoid)
        )
        grad_bias = grad_output.sum().reshape_as(bias)
        return grad_x, grad_bias, None


swish_bias_function = SwishBiasFunction.apply
