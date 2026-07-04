# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fully parameterized learnable neuron (Jolivet et al. 2006 threshold models)

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LearnableNeuronModel:
    """Fully parameterized learnable neuron.

    V[t+1] = alpha * V[t] + beta * I[t] + gamma * f(V[t])
    where alpha, beta, gamma are trainable scalars and f is a
    learnable activation (here sigmoid) — a trainable generalisation of
    the simple threshold models fitted to cortical recordings by Jolivet et al.

    Reference: Jolivet, Rauch, Lüscher & Gerstner (2006). Predicting spike
    timing of neocortical pyramidal neurons by simple threshold models.
    J Comput Neurosci 21:35-49.
    """

    v: float = 0.0
    alpha: float = 0.9
    beta: float = 0.1
    gamma: float = 0.05
    v_threshold: float = 1.0
    v_reset: float = 0.0
    f_slope: float = 5.0  # sigmoid steepness
    f_shift: float = 0.5  # sigmoid center

    def __post_init__(self) -> None:
        for name in ("v", "alpha", "beta", "gamma", "v_reset", "f_shift"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("v_threshold", "f_slope"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        f_v = self._sigmoid(self.f_slope * (self.v - self.f_shift))
        self.v = self.alpha * self.v + self.beta * current + self.gamma * f_v
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = 0.0
