# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive Threshold MoE Neuron (SpikingBrain)

"""Adaptive threshold neuron with integer spike counts for Mixture-of-Experts.

Implements the SpikingBrain activation function (arXiv:2509.05276v2, Sep 2025)
which produces *integer* spike counts rather than binary spikes, enabling
direct use in transformer MoE layers with minimal information loss.

Equations (SpikingBrain-1.0, Section 3.2):

    V_th = (1/k) · mean(|x|)          — adaptive threshold from input statistics
    v[t+1] = v[t] + x[t+1]            — membrane integration
    s_INT = round(v / V_th)            — integer spike count
    v <- v - V_th · s                  — soft reset (residual preserved)

Time-collapsed mode (single-step inference):

    s_INT = round(x / V_th)

Parameter *k* controls the firing rate / sparsity trade-off:
higher k -> lower threshold -> more spikes -> less sparsity.

Reference: SpikingBrain-1.0, arXiv:2509.05276v2, September 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdaptiveThresholdMoENeuron:
    """Adaptive threshold MoE spiking neuron (SpikingBrain).

    Parameters
    ----------
    k : float
        Firing rate control (higher k -> lower threshold -> more spikes).
        Default: 4.0 (SpikingBrain recommended).
    ema_alpha : float
        EMA decay for running mean of |input|. Default: 0.1.
    """

    k: float = 4.0
    ema_alpha: float = 0.1

    v: float = field(default=0.0, repr=False)
    v_th: float = field(default=1.0, repr=False)
    _mean_abs_x: float = field(default=0.0, repr=False)

    def step(self, current: float) -> int:
        """Advance one timestep. Returns integer spike count (>= 0).

        Implements: V_th = mean(|x|)/k, v += x, s = round(v/V_th), v -= V_th*s.
        """
        self._mean_abs_x = (1.0 - self.ema_alpha) * self._mean_abs_x + self.ema_alpha * abs(current)
        self.v_th = self._mean_abs_x / self.k if self._mean_abs_x > 1e-12 else 1.0
        self.v += current
        s_int = round(self.v / self.v_th) if self.v_th > 1e-12 else 0
        if s_int != 0:
            self.v -= self.v_th * s_int
        return max(s_int, 0)

    def step_collapsed(self, activation: float) -> int:
        """Time-collapsed single-step: s_INT = round(x / V_th)."""
        self._mean_abs_x = (1.0 - self.ema_alpha) * self._mean_abs_x + self.ema_alpha * abs(
            activation
        )
        self.v_th = self._mean_abs_x / self.k if self._mean_abs_x > 1e-12 else 1.0
        return max(round(activation / self.v_th), 0)

    def sparsity(self) -> float:
        """Current activation sparsity (1.0 if below threshold, 0.0 if firing)."""
        return 1.0 if abs(self.v) < self.v_th else 0.0

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.v = 0.0
        self._mean_abs_x = 0.0
        self.v_th = 1.0
