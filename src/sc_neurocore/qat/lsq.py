# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Learned Step Size Quantization (LSQ)

"""Learned Step Size Quantization for quantisation-aware training.

LSQ makes the quantiser's step size ``s`` a trainable parameter learned jointly
with the weights, rather than a fixed function of the running weight range. The
forward pass quantises ``v`` onto the integer grid ``[qmin, qmax]`` at step
``s``; the backward pass propagates a gradient to ``s`` itself:

    ∂v̂/∂s = round(v/s) - v/s        for qmin < v/s < qmax
    ∂v̂/∂s = qmin                     for v/s <= qmin
    ∂v̂/∂s = qmax                     for v/s >= qmax

and a straight-through gradient of 1 to ``v`` inside the clip range (0 outside).
The step-size gradient is scaled by ``1 / sqrt(qmax * n)`` (``n`` = elements per
step) so its magnitude matches the weight gradients during joint optimisation.
The step is initialised to ``2 * mean(|v|) / sqrt(qmax)``.

A single scalar step gives per-tensor quantisation; one step per output channel
gives per-channel quantisation, which pairs naturally with the per-channel
observers in :mod:`sc_neurocore.qat.observers`.

Reference: Esser et al. 2020 — "Learned Step Size Quantization" (ICLR).
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn as nn


def _sum_to(grad: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Reduce ``grad`` by summation to broadcast-target ``shape``.

    Sums every axis where the target shape has extent 1 (or is absent),
    inverting the broadcast that produced ``grad`` from a parameter of
    ``shape``. Used to fold a per-element step-size gradient back onto a scalar
    or per-channel step.

    Parameters
    ----------
    grad : torch.Tensor
        Per-element gradient.
    shape : tuple of int
        Target parameter shape.

    Returns
    -------
    torch.Tensor
        Gradient reduced to ``shape``.
    """
    while grad.dim() > len(shape):
        grad = grad.sum(dim=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(dim=axis, keepdim=True)
    return grad


class _LSQQuantize(torch.autograd.Function):
    """Autograd op implementing the LSQ forward quantiser and step gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        v: torch.Tensor,
        step: torch.Tensor,
        qmin: int,
        qmax: int,
        grad_scale: float,
    ) -> torch.Tensor:
        """Quantise ``v`` at learned ``step`` onto ``[qmin, qmax]``."""
        ctx.save_for_backward(v, step)
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.grad_scale = grad_scale
        v_s = v / step
        v_clip = v_s.clamp(qmin, qmax)
        v_bar = torch.round(v_clip)
        return v_bar * step

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        """Return gradients w.r.t. ``v`` (STE-in-range) and ``step`` (LSQ)."""
        v, step = ctx.saved_tensors
        qmin: int = ctx.qmin
        qmax: int = ctx.qmax
        grad_scale: float = ctx.grad_scale

        v_s = v / step
        below = v_s < qmin
        above = v_s > qmax
        inside = ~(below | above)

        # Straight-through gradient to the value: identity inside the grid.
        grad_v = grad_output * inside

        # LSQ gradient to the step size.
        grad_step_elem = torch.where(
            inside,
            torch.round(v_s) - v_s,
            torch.where(
                below,
                torch.full_like(v_s, float(qmin)),
                torch.full_like(v_s, float(qmax)),
            ),
        )
        grad_step_full = grad_output * grad_step_elem * grad_scale
        grad_step = _sum_to(grad_step_full, tuple(step.shape))
        return grad_v, grad_step, None, None, None


class LSQQuantizer(nn.Module):
    """Learned-step-size fake quantiser for signed weights.

    Parameters
    ----------
    n_bits : int
        Quantiser bit width (``>= 2``). The signed grid is
        ``[-2**(n_bits-1), 2**(n_bits-1) - 1]``.
    per_channel : bool
        Learn one step per channel along ``ch_axis`` instead of a single
        scalar step.
    ch_axis : int
        Channel axis used when ``per_channel`` is set.
    num_channels : int, optional
        Channel count; required when ``per_channel`` is set.

    Attributes
    ----------
    step : torch.nn.Parameter
        The learned step size(s). Lazily initialised from the first input.
    """

    step: nn.Parameter
    _initialized: torch.Tensor

    def __init__(
        self,
        n_bits: int = 8,
        *,
        per_channel: bool = False,
        ch_axis: int = 0,
        num_channels: int | None = None,
    ) -> None:
        super().__init__()
        if n_bits < 2:
            raise ValueError(f"n_bits must be >= 2, got {n_bits}")
        if per_channel and num_channels is None:
            raise ValueError("per_channel quantisation requires num_channels")
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.qmin = -(1 << (n_bits - 1))
        self.qmax = (1 << (n_bits - 1)) - 1
        if per_channel:
            assert num_channels is not None  # validated above
            init = torch.ones(num_channels)
        else:
            init = torch.ones(())
        self.step = nn.Parameter(init)
        self.register_buffer("_initialized", torch.zeros((), dtype=torch.bool))

    def _init_step_from(self, x: torch.Tensor) -> None:
        """Initialise the step to ``2*mean(|x|)/sqrt(qmax)`` (Esser et al. 2020)."""
        with torch.no_grad():
            if self.per_channel:
                axis = self.ch_axis % x.dim()
                moved = x.detach().movedim(axis, 0).reshape(x.shape[axis], -1)
                mean_abs = moved.abs().mean(dim=1)
            else:
                mean_abs = x.detach().abs().mean()
            self.step.copy_(2.0 * mean_abs / math.sqrt(self.qmax))
            self._initialized.fill_(True)

    def _broadcast_step(self, ndim: int) -> torch.Tensor:
        """Reshape the per-channel step for broadcasting over ``ndim`` dims."""
        if not self.per_channel:
            return self.step
        axis = self.ch_axis % ndim
        shape = [1] * ndim
        shape[axis] = -1
        return self.step.reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantise ``x`` at the learned step, with the LSQ gradient.

        Parameters
        ----------
        x : torch.Tensor
            Full-precision weights to quantise.

        Returns
        -------
        torch.Tensor
            The quantised-dequantised weights, differentiable in both ``x`` and
            the step size.
        """
        if not bool(self._initialized):
            self._init_step_from(x)
        n_elements = x.numel() // self.step.numel()
        grad_scale = 1.0 / math.sqrt(self.qmax * max(n_elements, 1))
        step = self._broadcast_step(x.dim()).clamp(min=1e-8)
        return cast(
            torch.Tensor,
            _LSQQuantize.apply(x, step, self.qmin, self.qmax, grad_scale),
        )

    def integer_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return integer codes and the step(s) for hardware export.

        Parameters
        ----------
        x : torch.Tensor
            Weights to encode at the learned step.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Integer codes clamped to the grid, and the per-tensor or
            per-channel step size(s).
        """
        step = self._broadcast_step(x.dim()).clamp(min=1e-8)
        codes = torch.round(x / step).clamp(self.qmin, self.qmax)
        return codes.to(torch.int32), self.step.detach()


class LSQLinear(nn.Module):
    """Linear layer whose weights are quantised by a learned step size.

    Per-channel (per output neuron) quantisation is the default, matching the
    granularity that recovers most of the accuracy lost to low-bit weights.

    Parameters
    ----------
    in_features, out_features : int
        Layer dimensions.
    n_bits : int
        Weight quantiser bit width.
    per_channel : bool
        Learn one step per output neuron (default) or a single scalar step.
    bias : bool
        Whether to include a full-precision bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        n_bits: int = 8,
        per_channel: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_quant = LSQQuantizer(
            n_bits,
            per_channel=per_channel,
            ch_axis=0,
            num_channels=out_features if per_channel else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer with LSQ-quantised weights."""
        w_q = self.weight_quant(self.linear.weight)
        return nn.functional.linear(x, w_q, self.linear.bias)

    def export_quantized(self) -> dict[str, Any]:
        """Export integer weights, the learned step(s), and the bias.

        Returns
        -------
        dict
            ``weight_int`` (int32 codes), ``step`` (per-tensor or per-channel),
            ``n_bits``, ``per_channel``, and optionally ``bias``.
        """
        codes, step = self.weight_quant.integer_weights(self.linear.weight.detach())
        result: dict[str, Any] = {
            "weight_int": codes,
            "step": step,
            "n_bits": self.weight_quant.n_bits,
            "per_channel": self.weight_quant.per_channel,
        }
        if self.linear.bias is not None:
            result["bias"] = self.linear.bias.detach()
        return result
