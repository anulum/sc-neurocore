# SPDX-License-Identifier: AGPL-3.0-or-later
"""Surrogate gradient functions for backprop through spike discontinuities.

Forward pass: Heaviside step function (hard threshold).
Backward pass: smooth surrogate gradient that approximates the Dirac delta.

All functions expect pre-shifted input x = v - threshold.
"""

from __future__ import annotations

import torch
from torch.autograd import Function


class _FastSigmoid(Function):
    """Zenke & Ganguli 2018, Eq. 11."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad = ctx.slope / (1.0 + ctx.slope * x.abs()) ** 2
        return grad_output * grad, None


class _SuperSpike(Function):
    """Zenke & Vogels 2021."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, beta: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad = 1.0 / (1.0 + ctx.beta * x.abs()) ** 2
        return grad_output * grad, None


class _ATan(Function):
    """Fang et al. 2021."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        grad = a / (2.0 * (1.0 + (torch.pi * a * x / 2.0) ** 2))
        return grad_output * grad, None


def fast_sigmoid(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    """Heaviside forward, fast-sigmoid backward."""
    return _FastSigmoid.apply(x, slope)


def superspike(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Heaviside forward, SuperSpike backward."""
    return _SuperSpike.apply(x, beta)


def atan_surrogate(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Heaviside forward, arctan backward."""
    return _ATan.apply(x, alpha)
