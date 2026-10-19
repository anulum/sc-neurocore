# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brunel & Hakim 1999 — Ornstein-Uhlenbeck driven IF

"""Integrate-and-fire neuron driven by Gaussian white noise drawn from NumPy's global generator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StochasticIFNeuron:
    """Brunel & Hakim 1999 — Ornstein-Uhlenbeck driven IF.

    Reference: Tuckwell, H.C. (1988). Introduction to Theoretical Neurobiology, Vol. 2. Cambridge Univ. Press.
    """

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    mu: float = 0.0
    sigma: float = 3.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Refuse non-finite voltages and drift, non-positive time constants and negative sigma."""
        for field in ("v", "v_rest", "v_reset", "v_threshold", "mu"):
            value = getattr(self, field)
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        for field in ("tau_m", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be positive and finite")
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError("sigma must be non-negative and finite")

    def step(self, current: float) -> int:
        """Advance one Euler-Maruyama step with one ``np.random.randn()`` draw; return the spike.

        Parameters
        ----------
        current : float
            Finite input current of this step.

        Returns
        -------
        int
            1 when the voltage reached ``v_threshold`` (and was reset), else 0.
        """
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        noise = self.sigma * np.sqrt(self.dt / self.tau_m) * np.random.randn()
        self.v += (-(self.v - self.v_rest) + self.mu + current) / self.tau_m * self.dt + noise
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        """Return the membrane voltage to ``v_rest``."""
        self.v = self.v_rest
