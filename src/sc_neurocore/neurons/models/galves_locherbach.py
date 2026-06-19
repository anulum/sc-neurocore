# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Galves-Löcherbach 2013 — stochastic point process neuron

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GalvesLocherbachNeuron:
    """Galves-Löcherbach 2013 — stochastic point process neuron.

    P(spike at t | history) = φ(V(t))
    V(t) = Σ w_j · spike_j(past) · decay + leak
    Purely probabilistic, no ODE.

    Reference: Galves, A. & Löcherbach, E. (2013). J. Stat. Phys. 151:896–921.
    """

    v: float = 0.0
    v_rest: float = 0.0
    decay: float = 0.95
    threshold_rate: float = 0.5
    steepness: float = 5.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        for field in ("v", "v_rest", "threshold_rate"):
            value = getattr(self, field)
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if not math.isfinite(self.decay) or not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be finite and within [0, 1]")
        if not math.isfinite(self.steepness) or self.steepness <= 0.0:
            raise ValueError("steepness must be positive and finite")
        if not math.isfinite(self.dt) or not 0.0 < self.dt <= 1.0:
            raise ValueError("dt must be finite and within (0, 1]")

    def _firing_prob(self) -> float:
        z = self.steepness * (self.v - self.threshold_rate)
        if z >= 0.0:
            tail = math.exp(-z)
            return 1.0 / (1.0 + tail)
        tail = math.exp(z)
        return tail / (1.0 + tail)

    def step(self, weighted_input: float) -> int:
        if not math.isfinite(weighted_input):
            raise ValueError("weighted_input must be finite")
        self.v = self.decay * self.v + weighted_input
        p = self._firing_prob()
        spike = 1 if np.random.random() < p * self.dt else 0
        if spike:
            self.v = self.v_rest
        return spike

    def reset(self) -> None:
        self.v = self.v_rest
