# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aihara Chaotic Map Neuron

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AiharaMapNeuron:
    """Aihara 1990 chaotic map neuron.

    2D discrete map exhibiting chaotic spiking dynamics. The fast variable
    x is driven by a sigmoidal self-feedback modulated by k_f, minus the
    slow recovery variable y.

    x(n+1) = k_f · x(n) · σ(x(n) + α) - y(n) + I
    y(n+1) = k_s · y(n) + δ · x(n)

    where σ(z) = 1 / (1 + exp(-z)).

    Reference: Aihara et al. (1990) Phys Lett A 144:333–340.
    """

    x: float = 0.0
    y: float = 0.0
    k_f: float = 0.7
    k_s: float = 0.95
    alpha: float = 2.0
    delta: float = 0.05
    x_threshold: float = 0.5

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        sigmoid = 1.0 / (1.0 + math.exp(-(self.x + self.alpha)))
        x_new = self.k_f * self.x * sigmoid - self.y + current
        y_new = self.k_s * self.y + self.delta * self.x

        self.x = max(-10.0, min(10.0, x_new))
        self.y = max(-10.0, min(10.0, y_new))

        if not math.isfinite(self.x):
            self.x = 0.0
        if not math.isfinite(self.y):
            self.y = 0.0

        return 1 if self.x >= self.x_threshold and x_prev < self.x_threshold else 0

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
