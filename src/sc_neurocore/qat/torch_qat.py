# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — PyTorch Quantization-Aware Training for SNNs

"""Quantization-aware training (QAT) for PyTorch SNN modules.

Wraps existing LIFCell with straight-through estimator (STE) weight
quantization. Supports 2, 4, 8, 16-bit weight precision.

During training: weights quantized in forward pass, full-precision
gradients flow through STE in backward pass.

At export: call export_quantized() to get integer weights at target bits.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, cast

import torch
import torch.nn as nn

from ..training.snn_modules import LIFCell, atan_surrogate  # type: ignore[attr-defined]


class _STEQuantize(torch.autograd.Function):
    """Straight-through estimator for uniform quantization."""

    @staticmethod
    def forward(ctx, x, n_bits, symmetric=True):  # type: ignore[no-untyped-def]
        n_levels = 2**n_bits
        if symmetric:
            abs_max = x.abs().max().clamp(min=1e-8)
            half = n_levels // 2 - 1
            scale = abs_max / half
            x_q = (x / scale).round().clamp(-half, half) * scale
        else:
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min).clamp(min=1e-8) / (n_levels - 1)
            x_q = ((x - x_min) / scale).round() * scale + x_min
        return x_q

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[no-untyped-def]
        # STE: pass gradient through unchanged
        return grad_output, None, None


def ste_quantize(x: torch.Tensor, n_bits: int, symmetric: bool = True) -> torch.Tensor:
    """Quantize tensor with straight-through estimator."""
    return cast(torch.Tensor, _STEQuantize.apply(x, n_bits, symmetric))


class QuantizedLinear(nn.Module):
    """Linear layer with STE weight quantization."""

    def __init__(self, in_features: int, out_features: int, n_bits: int = 8, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.n_bits = n_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = ste_quantize(self.linear.weight, self.n_bits)
        out = nn.functional.linear(x, w_q, self.linear.bias)
        return out

    def export_quantized(self) -> dict[str, Any]:
        """Export integer weights at target precision."""
        w = self.linear.weight.detach()
        abs_max = w.abs().max().clamp(min=1e-8)
        half = 2 ** (self.n_bits - 1) - 1
        scale = abs_max / half
        w_int = (w / scale).round().clamp(-half, half).to(torch.int8)
        result = {"weight_int": w_int, "scale": scale.item(), "n_bits": self.n_bits}
        if self.linear.bias is not None:
            result["bias"] = self.linear.bias.detach()
        return result


class QuantizedLIFNet(nn.Module):
    """Feedforward SNN with quantized weights for QAT.

    Drop-in replacement for SpikingNet with configurable bit precision.

    Example
    -------
    >>> net = QuantizedLIFNet(784, 128, 10, n_bits=4)
    >>> x = torch.randn(25, 32, 784)  # (T, batch, features)
    >>> spikes, mem = net(x)
    >>> spikes.shape
    torch.Size([32, 10])
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 2,
        n_bits: int = 8,
        beta: float = 0.9,
        surrogate_fn: Callable[..., torch.Tensor] = atan_surrogate,
    ):
        super().__init__()
        self.n_output = n_output
        self.n_bits = n_bits

        sizes = [n_input] + [n_hidden] * n_layers + [n_output]
        self.linears = nn.ModuleList(
            QuantizedLinear(sizes[i], sizes[i + 1], n_bits=n_bits) for i in range(len(sizes) - 1)
        )
        self.lifs = nn.ModuleList(
            LIFCell(beta=beta, surrogate_fn=surrogate_fn) for _ in range(len(sizes) - 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (T, batch, n_input). Returns (spike_counts, membrane_acc)."""
        T, batch, _ = x.shape
        device = x.device
        v = [
            torch.zeros(batch, cast(QuantizedLinear, lin).linear.out_features, device=device)
            for lin in self.linears
        ]

        spike_sum = torch.zeros(batch, self.n_output, device=device)
        mem_sum = torch.zeros(batch, self.n_output, device=device)

        for t in range(T):
            h = x[t]
            for i in range(len(self.linears)):
                h = cast(QuantizedLinear, self.linears[i])(h)
                spike, v[i] = cast(LIFCell, self.lifs[i])(h, v[i])
                h = spike
            spike_sum = spike_sum + spike
            mem_sum = mem_sum + v[-1]

        return spike_sum, mem_sum

    def export_quantized(self) -> list[dict[str, Any]]:
        """Export all layers as quantized integer weights."""
        return [cast(QuantizedLinear, lin).export_quantized() for lin in self.linears]

    def effective_bits(self) -> float:
        """Average effective bits across all weights (for reporting)."""
        total_params = 0
        total_bits = 0
        for lin in self.linears:
            n = cast(QuantizedLinear, lin).linear.weight.numel()
            total_params += n
            total_bits += n * self.n_bits
        return total_bits / max(total_params, 1)


class SCAwareLinear(nn.Module):
    """Linear layer with SC noise injection during training.

    During training: injects Gaussian noise with std = sqrt(p*(1-p)/L)
    to simulate bitstream variance. Weights clamped to [-1, 1].

    During eval: no noise, standard linear.
    """

    def __init__(
        self, in_features: int, out_features: int, bitstream_length: int = 256, bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bitstream_length = bitstream_length
        # Clamp weights to [-1, 1] at init
        with torch.no_grad():
            self.linear.weight.clamp_(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp weights to bipolar range during forward
        w = self.linear.weight.clamp(-1.0, 1.0)

        if self.training:
            # SC noise: std = sqrt(p * (1-p) / L) where p = (w + 1) / 2
            p = (w + 1.0) / 2.0
            sc_variance = p * (1.0 - p) / self.bitstream_length
            noise = torch.randn_like(w) * sc_variance.sqrt()
            w = w + noise

        return nn.functional.linear(x, w, self.linear.bias)


class SCAwareLIFNet(nn.Module):
    """SNN with SC-aware training: noise injection + weight clamping.

    Trains the model to be robust to stochastic computing bitstream
    variance. Weights are constrained to [-1, 1] (bipolar SC range).

    Example
    -------
    >>> net = SCAwareLIFNet(784, 128, 10, bitstream_length=256)
    >>> x = torch.randn(25, 32, 784)
    >>> spikes, mem = net(x)
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 2,
        bitstream_length: int = 256,
        beta: float = 0.9,
        surrogate_fn: Callable[..., torch.Tensor] = atan_surrogate,
    ):
        super().__init__()
        self.n_output = n_output
        self.bitstream_length = bitstream_length

        sizes = [n_input] + [n_hidden] * n_layers + [n_output]
        self.linears = nn.ModuleList(
            SCAwareLinear(sizes[i], sizes[i + 1], bitstream_length=bitstream_length)
            for i in range(len(sizes) - 1)
        )
        self.lifs = nn.ModuleList(
            LIFCell(beta=beta, surrogate_fn=surrogate_fn) for _ in range(len(sizes) - 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (T, batch, n_input). Returns (spike_counts, membrane_acc)."""
        T, batch, _ = x.shape
        device = x.device
        v = [
            torch.zeros(batch, cast(SCAwareLinear, lin).linear.out_features, device=device)
            for lin in self.linears
        ]

        spike_sum = torch.zeros(batch, self.n_output, device=device)
        mem_sum = torch.zeros(batch, self.n_output, device=device)

        for t in range(T):
            h = x[t]
            for i in range(len(self.linears)):
                h = cast(SCAwareLinear, self.linears[i])(h)
                spike, v[i] = cast(LIFCell, self.lifs[i])(h, v[i])
                h = spike
            spike_sum = spike_sum + spike
            mem_sum = mem_sum + v[-1]

        return spike_sum, mem_sum

    def export_bipolar_weights(self) -> list[dict[str, Any]]:
        """Export weights clamped to [-1, 1] for bipolar SC deployment."""
        layers = []
        for lin in self.linears:
            lin_typed = cast(SCAwareLinear, lin)
            w = lin_typed.linear.weight.detach().clamp(-1.0, 1.0)
            entry = {"weight": w.cpu().numpy()}
            if lin_typed.linear.bias is not None:
                entry["bias"] = lin_typed.linear.bias.detach().cpu().numpy()
            layers.append(entry)
        return layers
