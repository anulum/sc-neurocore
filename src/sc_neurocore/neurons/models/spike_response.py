# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpikeResponseNeuron:
    """Spike Response Model (SRM0) — kernel-based, no ODEs.

    v(t) = η(t - t_last) + Σ κ(t - t_in) · w
    Spike when v(t) ≥ threshold.
    Gerstner 1995.
    """

    v: float = 0.0
    v_threshold: float = 1.0
    tau_eta: float = 10.0
    tau_kappa: float = 5.0
    eta_reset: float = -5.0
    time_since_spike: float = 1000.0
    dt: float = 1.0

    def step(self, weighted_input: float) -> int:
        # Refractory kernel (spike afterpotential)
        eta = (
            self.eta_reset * np.exp(-self.time_since_spike / self.tau_eta)
            if self.time_since_spike < 100.0
            else 0.0
        )
        # Input kernel
        kappa = weighted_input * (1.0 - np.exp(-self.dt / self.tau_kappa))
        self.v = eta + kappa
        self.time_since_spike += self.dt

        if self.v >= self.v_threshold:
            self.time_since_spike = 0.0
            self.v = 0.0
            return 1
        return 0

    def reset(self):
        self.v = 0.0
        self.time_since_spike = 1000.0
