# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Jahns et al. 2025 — fully parameterized learnable neuron

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LearnableNeuronModel:
    """Jahns et al. 2025 — fully parameterized learnable neuron.

    V[t+1] = alpha * V[t] + beta * I[t] + gamma * f(V[t])
    where alpha, beta, gamma are trainable scalars and f is a
    learnable activation (here sigmoid).
    """

    v: float = 0.0
    alpha: float = 0.9
    beta: float = 0.1
    gamma: float = 0.05
    v_threshold: float = 1.0
    v_reset: float = 0.0
    f_slope: float = 5.0  # sigmoid steepness
    f_shift: float = 0.5  # sigmoid center

    def step(self, current: float) -> int:
        f_v = 1.0 / (1.0 + np.exp(-self.f_slope * (self.v - self.f_shift)))
        self.v = self.alpha * self.v + self.beta * current + self.gamma * f_v
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = 0.0
