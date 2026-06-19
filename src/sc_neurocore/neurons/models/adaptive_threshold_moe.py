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

import math
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

    def __post_init__(self) -> None:
        if not math.isfinite(self.k) or self.k <= 0.0:
            raise ValueError("k must be finite and positive")
        if not math.isfinite(self.ema_alpha) or not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be finite and in (0, 1]")
        if not math.isfinite(self.v):
            raise ValueError("v must be finite")
        if not math.isfinite(self.v_th) or self.v_th <= 0.0:
            raise ValueError("v_th must be finite and positive")
        if not math.isfinite(self._mean_abs_x) or self._mean_abs_x < 0.0:
            raise ValueError("_mean_abs_x must be finite and non-negative")

    def step(self, current: float) -> int:
        """Advance one timestep. Returns integer spike count (>= 0).

        Implements: V_th = mean(|x|)/k, v += x, s = round(v/V_th), v -= V_th*s.
        """
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        next_mean_abs_x = (1.0 - self.ema_alpha) * self._mean_abs_x + self.ema_alpha * abs(current)
        next_v_th = self._threshold_from_mean(next_mean_abs_x)
        next_v = self.v + current
        if not math.isfinite(next_v):
            raise ValueError("soft reset residual must remain finite")
        ratio = next_v / next_v_th if next_v_th > 1e-12 else 0.0
        if not math.isfinite(ratio):
            raise ValueError("adaptive threshold ratio must remain finite")
        s_int = max(round(ratio), 0)
        residual = next_v - next_v_th * s_int if s_int != 0 else next_v
        if not math.isfinite(residual):
            raise ValueError("soft reset residual must remain finite")

        self._mean_abs_x = next_mean_abs_x
        self.v_th = next_v_th
        self.v = residual
        return s_int

    def step_collapsed(self, activation: float) -> int:
        """Time-collapsed single-step: s_INT = round(x / V_th)."""
        if not math.isfinite(activation):
            raise ValueError("activation must be finite")
        self._validate_runtime_state()

        next_mean_abs_x = (1.0 - self.ema_alpha) * self._mean_abs_x + self.ema_alpha * abs(
            activation
        )
        next_v_th = self._threshold_from_mean(next_mean_abs_x)
        ratio = activation / next_v_th
        if not math.isfinite(ratio):
            raise ValueError("adaptive threshold ratio must remain finite")
        spike_count = max(round(ratio), 0)
        self._mean_abs_x = next_mean_abs_x
        self.v_th = next_v_th
        return spike_count

    def sparsity(self) -> float:
        """Current activation sparsity (1.0 if below threshold, 0.0 if firing)."""
        self._validate_runtime_state()
        return 1.0 if abs(self.v) < self.v_th else 0.0

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.v = 0.0
        self._mean_abs_x = 0.0
        self.v_th = 1.0

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError("runtime membrane state must be finite")
        if not math.isfinite(self.v_th) or self.v_th <= 0.0:
            raise ValueError("runtime threshold state must be finite and positive")
        if not math.isfinite(self._mean_abs_x) or self._mean_abs_x < 0.0:
            raise ValueError("runtime mean absolute input state must be finite and non-negative")

    def _threshold_from_mean(self, mean_abs_x: float) -> float:
        if not math.isfinite(mean_abs_x) or mean_abs_x < 0.0:
            raise ValueError("adaptive threshold mean must remain finite and non-negative")
        threshold = mean_abs_x / self.k if mean_abs_x > 1e-12 else 1.0
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("adaptive threshold must remain finite and positive")
        return threshold
