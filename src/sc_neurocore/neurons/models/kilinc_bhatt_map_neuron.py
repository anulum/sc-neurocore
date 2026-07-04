# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Nagumo-Sato / Aihara sigmoid map neuron with dynamic threshold

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KilincBhattMapNeuron:
    """Nagumo-Sato / Aihara sigmoid map neuron with a dynamic threshold.

    Minimal 2D map with built-in spike frequency adaptation via a slow
    threshold variable. Designed for efficient hardware implementation
    while retaining biologically relevant dynamics.

    x(n+1) = -x(n) + k · σ(4·(x(n) - θ(n))) + I
    θ(n+1) = β · θ(n) + γ · H(x(n) - θ_spike)

    where σ(z) = 1 / (1 + exp(-z)) and H() is Heaviside. This is the
    Nagumo-Sato (1972) discrete-time neuron with an accumulated dynamic
    threshold, using the Aihara, Takabe & Toyoda (1990) sigmoid firing in
    place of the hard Heaviside so that spiking is graded rather than
    all-or-nothing.

    References: Nagumo & Sato (1972) Kybernetik 10:155-164;
    Aihara, Takabe & Toyoda (1990) Phys Lett A 144:333-340.
    """

    x: float = 0.0
    theta: float = 0.0
    k: float = 1.5
    beta: float = 0.95
    gamma: float = 0.3
    theta_spike: float = 0.8
    x_threshold: float = 0.8

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        sig = 1.0 / (1.0 + math.exp(-(self.x - self.theta) * 4.0))
        x_new = -self.x + self.k * sig + current
        spiked = 1.0 if self.x >= self.theta_spike else 0.0
        theta_new = self.beta * self.theta + self.gamma * spiked

        self.x = max(-5.0, min(5.0, x_new))
        self.theta = max(-5.0, min(5.0, theta_new))

        if not math.isfinite(self.x):
            self.x = 0.0
        if not math.isfinite(self.theta):
            self.theta = 0.0

        return 1 if self.x >= self.x_threshold and x_prev < self.x_threshold else 0

    def reset(self) -> None:
        self.x = 0.0
        self.theta = 0.0
