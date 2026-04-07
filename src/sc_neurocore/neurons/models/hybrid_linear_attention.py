# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hybrid Linear Attention Neuron (SpikingBrain)

"""Hybrid linear attention neuron for spiking environments.

Combines local windowed attention with linear (kernel-based) global attention,
achieving near-linear training complexity O(L) instead of O(L**2).
Inspired by SpikingBrain's hybrid attention architecture.

The neuron accumulates spike-weighted keys and values via a recurrent
state S, avoiding the quadratic attention matrix:

    S(t+1) = lambda * S(t) + phi(k_t) (x) v_t
    output  = phi(q_t)^T S(t)

where phi is an elu+1 feature map: phi(x) = x + 1 if x > 0, else exp(x).

Reference: SpikingBrain hybrid attention, arXiv:2509.05276v2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class HybridLinearAttentionNeuron:
    """Hybrid linear attention spiking neuron.

    Parameters
    ----------
    dim : int
        Dimension of recurrent KV state. Default: 16.
    lambda_decay : float
        Exponential decay for recurrent state. Default: 0.95.
    window_size : int
        Sliding window size for local attention. Default: 16.
    """

    dim: int = 16
    lambda_decay: float = 0.95
    window_size: int = 16
    dt: float = 1.0

    v: float = field(default=0.0, repr=False)
    _state_kv: list[float] = field(default_factory=list, repr=False)
    _window_buf: list[float] = field(default_factory=list, repr=False)
    _window_idx: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self._state_kv:
            self._state_kv = [0.0] * self.dim
        if not self._window_buf:
            self._window_buf = [0.0] * self.window_size

    @staticmethod
    def _phi(x: float) -> float:
        """Feature map: elu(x) + 1."""
        return x + 1.0 if x > 0.0 else math.exp(x)

    def step_qkv(self, query: float, key: float, value: float) -> float:
        """Step with explicit query, key, value (scalar projections).

        Returns combined global + local attention output.
        """
        phi_q = self._phi(query)
        phi_k = self._phi(key)

        for i in range(self.dim):
            self._state_kv[i] *= self.lambda_decay
        idx = int(abs(phi_k) * self.dim) % self.dim
        self._state_kv[idx] += phi_k * value

        global_out = phi_q * self._state_kv[idx]

        self._window_buf[self._window_idx % self.window_size] = value
        self._window_idx += 1
        local_out = sum(self._window_buf) / self.window_size

        self.v = 0.5 * global_out + 0.5 * local_out
        return self.v

    def step(self, current: float) -> int:
        """Simple step (input treated as combined qkv). Returns spike (0 or 1)."""
        out = self.step_qkv(current, current, current)
        return 1 if out > 1.0 else 0

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.v = 0.0
        self._state_kv = [0.0] * self.dim
        self._window_buf = [0.0] * self.window_size
        self._window_idx = 0
