# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differentiable SNN layers with surrogate gradient support.

Train in float with PyTorch autograd, deploy to SC bitstreams via to_sc_weights().
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from .surrogate import atan_surrogate


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
        surrogate_fn: Callable = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        if learn_beta:
            # Store log-odds so sigmoid maps to (0, 1)
            self._beta_logit = nn.Parameter(torch.tensor(_logit(beta)))
        else:
            self.register_buffer("_beta_val", torch.tensor(beta))
        self._learn_beta = learn_beta

        if learn_threshold:
            # Store log so exp maps to (0, inf)
            self._threshold_log = nn.Parameter(torch.tensor(float(threshold)).log())
        else:
            self.register_buffer("_threshold_val", torch.tensor(threshold))
        self._learn_threshold = learn_threshold

        self.surrogate_fn = surrogate_fn

    @property
    def beta(self) -> torch.Tensor:
        return self._beta_logit.sigmoid() if self._learn_beta else self._beta_val

    @property
    def threshold(self) -> torch.Tensor:
        return self._threshold_log.exp() if self._learn_threshold else self._threshold_val

    def forward(self, current: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_next = self.beta * v + current
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike.detach() * self.threshold
        return spike, v_next


class IFCell(nn.Module):
    """Integrate-and-Fire (no leak, beta=1).

    Simplest spiking model: v[t] = v[t-1] + I[t], fire when v >= threshold.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        surrogate_fn: Callable = atan_surrogate,
    ):
        super().__init__()
        self.register_buffer("_threshold", torch.tensor(threshold))
        self.surrogate_fn = surrogate_fn

    def forward(self, current: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_next = v + current
        spike = self.surrogate_fn(v_next - self._threshold)
        v_next = v_next - spike.detach() * self._threshold
        return spike, v_next


class SynapticCell(nn.Module):
    """Dual-exponential synaptic LIF. Two state variables: synapse current + membrane.

    i_syn[t] = alpha * i_syn[t-1] + I[t]
    v[t] = beta * v[t-1] + i_syn[t]
    """

    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 0.8,
        threshold: float = 1.0,
        surrogate_fn: Callable = atan_surrogate,
        learn_beta: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        if learn_beta:
            self._beta_logit = nn.Parameter(torch.tensor(_logit(beta)))
        else:
            self.register_buffer("_beta_val", torch.tensor(beta))
        self._learn_beta = learn_beta
        self.register_buffer("_threshold", torch.tensor(threshold))
        self.surrogate_fn = surrogate_fn

    @property
    def beta(self) -> torch.Tensor:
        return self._beta_logit.sigmoid() if self._learn_beta else self._beta_val

    def forward(
        self, current: torch.Tensor, i_syn: torch.Tensor, v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (spike, i_syn_next, v_next)."""
        i_syn_next = self.alpha * i_syn + current
        v_next = self.beta * v + i_syn_next
        spike = self.surrogate_fn(v_next - self._threshold)
        v_next = v_next - spike.detach() * self._threshold
        return spike, i_syn_next, v_next


class ALIFCell(nn.Module):
    """Adaptive LIF. Bellec et al. 2020.

    Threshold adapts based on recent spiking: theta[t] = theta_0 + beta_adapt * a[t]
    where a[t] = rho * a[t-1] + spike[t-1].
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        rho: float = 0.99,
        beta_adapt: float = 1.8,
        surrogate_fn: Callable = atan_surrogate,
    ):
        super().__init__()
        self.beta = beta
        self.threshold_0 = threshold
        self.rho = rho
        self.beta_adapt = beta_adapt
        self.surrogate_fn = surrogate_fn

    def forward(
        self, current: torch.Tensor, v: torch.Tensor, a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (spike, v_next, a_next)."""
        v_next = self.beta * v + current
        theta = self.threshold_0 + self.beta_adapt * a
        spike = self.surrogate_fn(v_next - theta)
        v_next = v_next - spike.detach() * theta
        a_next = self.rho * a + spike.detach()
        return spike, v_next, a_next


class RecurrentLIFCell(nn.Module):
    """LIF with trainable recurrent weights."""

    def __init__(
        self,
        n_neurons: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_fn: Callable = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.lif = LIFCell(beta, threshold, surrogate_fn, learn_beta, learn_threshold)
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
        surrogate_fn: Callable = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.n_output = n_output
        sizes = [n_input] + [n_hidden] * n_layers + [n_output]
        self.linears = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )
        self.lifs = nn.ModuleList(
            LIFCell(
                beta=beta, surrogate_fn=surrogate_fn,
                learn_beta=learn_beta, learn_threshold=learn_threshold,
            )
            for _ in range(len(sizes) - 1)
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


class ConvSpikingNet(nn.Module):
    """Convolutional SNN for image classification.

    Conv2d(1,32,5)→LIF→AvgPool→Conv2d(32,64,5)→LIF→AvgPool→Flatten→Linear→LIF→Linear→LIF
    """

    def __init__(
        self,
        n_output: int = 10,
        beta: float = 0.9,
        surrogate_fn: Callable = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.n_output = n_output
        lif_kw = dict(
            beta=beta, surrogate_fn=surrogate_fn,
            learn_beta=learn_beta, learn_threshold=learn_threshold,
        )
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.lif1 = LIFCell(**lif_kw)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.lif2 = LIFCell(**lif_kw)
        self.pool2 = nn.AvgPool2d(2)
        # MNIST: 28→conv5→24→pool2→12→conv5→8→pool2→4, so 64*4*4=1024
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.lif3 = LIFCell(**lif_kw)
        self.fc2 = nn.Linear(128, n_output)
        self.lif4 = LIFCell(**lif_kw)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (T, batch, 1, 28, 28). Returns (spike_counts, membrane_acc)."""
        T, batch = x.shape[:2]
        device = x.device

        v1 = torch.zeros(batch, 32, 24, 24, device=device)
        v2 = torch.zeros(batch, 64, 8, 8, device=device)
        v3 = torch.zeros(batch, 128, device=device)
        v4 = torch.zeros(batch, self.n_output, device=device)

        spike_sum = torch.zeros(batch, self.n_output, device=device)
        mem_sum = torch.zeros(batch, self.n_output, device=device)

        for t in range(T):
            h = self.conv1(x[t])
            spk, v1 = self.lif1(h, v1)
            h = self.pool1(spk)

            h = self.conv2(h)
            spk, v2 = self.lif2(h, v2)
            h = self.pool2(spk)

            h = h.flatten(1)
            h = self.fc1(h)
            spk, v3 = self.lif3(h, v3)

            h = self.fc2(spk)
            spk, v4 = self.lif4(h, v4)

            spike_sum = spike_sum + spk
            mem_sum = mem_sum + v4

        return spike_sum, mem_sum

    def to_sc_weights(self) -> List[torch.Tensor]:
        """Export weight matrices normalized to [0,1] for SC bitstream deployment."""
        weights = []
        for mod in [self.conv1, self.conv2, self.fc1, self.fc2]:
            w = mod.weight.detach().flatten(1) if isinstance(mod, nn.Conv2d) else mod.weight.detach()
            w_min, w_max = w.min(), w.max()
            if w_max > w_min:
                w = (w - w_min) / (w_max - w_min)
            else:
                w = torch.zeros_like(w)
            weights.append(w)
        return weights


def _logit(p: float) -> float:
    """Inverse sigmoid: logit(p) = log(p / (1 - p))."""
    import math
    p = max(min(p, 1.0 - 1e-7), 1e-7)
    return math.log(p / (1.0 - p))
