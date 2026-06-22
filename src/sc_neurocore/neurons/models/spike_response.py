# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike Response Model (SRM0) — kernel-based, no ODEs

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SpikeResponseNeuron:
    """Spike Response Model (SRM0) — kernel-based, no ODEs.

    v(t) = η(t - t_last) + Σ κ(t - t_in) · w
    Spike when v(t) ≥ threshold.
    Gerstner 1995.

    Reference: Gerstner, W. (1995). Phys. Rev. E 51:738–758.
    """

    v: float = 0.0
    v_threshold: float = 1.0
    tau_eta: float = 10.0
    tau_kappa: float = 5.0
    eta_reset: float = -5.0
    time_since_spike: float = 1000.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        for name in ("v", "v_threshold", "eta_reset"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("tau_eta", "tau_kappa", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.time_since_spike) or self.time_since_spike < 0:
            raise ValueError("time_since_spike must be finite and non-negative")

    def step(self, weighted_input: float) -> int:
        if not math.isfinite(weighted_input):
            raise ValueError("weighted_input must be finite")

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

    def reset(self) -> None:
        self.v = 0.0
        self.time_since_spike = 1000.0
