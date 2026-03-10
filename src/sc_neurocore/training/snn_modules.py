# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differentiable SNN layers with surrogate gradient support.

Train in float with PyTorch autograd, deploy to SC bitstreams via to_sc_weights().
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from .surrogate import fast_sigmoid


class LIFCell(nn.Module):
    """Single-step Leaky Integrate-and-Fire with surrogate backward.

    v[t] = beta * v[t-1] + I[t]
    spike[t] = H(v[t] - threshold)
    v[t] -= spike[t] * threshold  (subtract reset)
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_fn: Callable = fast_sigmoid,
    ):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.surrogate_fn = surrogate_fn

    def forward(self, current: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_next = self.beta * v + current
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike * self.threshold
        return spike, v_next


class RecurrentLIFCell(nn.Module):
    """LIF with trainable recurrent weights."""

    def __init__(
        self,
        n_neurons: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_fn: Callable = fast_sigmoid,
    ):
        super().__init__()
        self.lif = LIFCell(beta, threshold, surrogate_fn)
        self.recurrent = nn.Linear(n_neurons, n_neurons, bias=False)
        nn.init.orthogonal_(self.recurrent.weight, gain=0.5)

    def forward(
        self,
        current: torch.Tensor,
        v: torch.Tensor,
        spike_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lif(current + self.recurrent(spike_prev), v)


class SpikingNet(nn.Module):
    """Multi-layer feedforward SNN for classification.

    Architecture: [Linear -> LIF] x (n_layers+1)
    Readout: spike count and membrane accumulation over T timesteps.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 2,
        beta: float = 0.9,
        surrogate_fn: Callable = fast_sigmoid,
    ):
        super().__init__()
        self.n_output = n_output
        sizes = [n_input] + [n_hidden] * n_layers + [n_output]
        self.linears = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )
        self.lifs = nn.ModuleList(
            LIFCell(beta=beta, surrogate_fn=surrogate_fn) for _ in range(len(sizes) - 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (T, batch, n_input). Returns (spike_counts, membrane_acc)."""
        T, batch, _ = x.shape
        device = x.device
        n_cells = len(self.lifs)
        v = [torch.zeros(batch, lin.out_features, device=device) for lin in self.linears]

        spike_sum = torch.zeros(batch, self.n_output, device=device)
        mem_sum = torch.zeros(batch, self.n_output, device=device)

        for t in range(T):
            h = x[t]
            for i in range(n_cells):
                h = self.linears[i](h)
                spike, v[i] = self.lifs[i](h, v[i])
                h = spike
            spike_sum = spike_sum + spike
            mem_sum = mem_sum + v[-1]

        return spike_sum, mem_sum

    def to_sc_weights(self) -> List[torch.Tensor]:
        """Export weight matrices normalized to [0,1] for SC bitstream deployment."""
        weights = []
        for lin in self.linears:
            w = lin.weight.detach()
            w_min, w_max = w.min(), w.max()
            if w_max > w_min:
                w = (w - w_min) / (w_max - w_min)
            else:
                w = torch.zeros_like(w)
            weights.append(w)
        return weights
