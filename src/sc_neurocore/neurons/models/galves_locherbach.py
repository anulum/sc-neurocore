# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Galves-Löcherbach 2013 — stochastic point process neuron

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GalvesLocherbachNeuron:
    """Galves-Löcherbach 2013 — stochastic point process neuron.

    P(spike at t | history) = φ(V(t))
    V(t) = Σ w_j · spike_j(past) · decay + leak
    Purely probabilistic, no ODE.
    """

    v: float = 0.0
    v_rest: float = 0.0
    decay: float = 0.95
    threshold_rate: float = 0.5
    steepness: float = 5.0
    dt: float = 1.0

    def _firing_prob(self):
        return 1.0 / (1.0 + np.exp(-self.steepness * (self.v - self.threshold_rate)))

    def step(self, weighted_input: float) -> int:
        self.v = self.decay * self.v + weighted_input
        p = self._firing_prob()
        spike = 1 if np.random.random() < p * self.dt else 0
        if spike:
            self.v = self.v_rest
        return spike

    def reset(self):
        self.v = self.v_rest
