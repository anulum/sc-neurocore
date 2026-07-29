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

    def __post_init__(self) -> None:
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for name in ("x", "theta", "k", "beta", "gamma", "theta_spike", "x_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)

        for name in ("x", "theta"):
            if not -5.0 <= getattr(self, name) <= 5.0:
                raise ValueError(f"{name} must be within [-5, 5]")
        if not 0.0 <= self.k <= 5.0:
            raise ValueError("k must be within [0, 5]")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be within [0, 1]")
        if not 0.0 <= self.gamma <= 2.0:
            raise ValueError("gamma must be within [0, 2]")
        for name in ("theta_spike", "x_threshold"):
            if not 0.0 <= getattr(self, name) <= 2.0:
                raise ValueError(f"{name} must be within [0, 2]")

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def step(self, current: float = 0.0) -> int:
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        self._validate_configuration()

        x_prev = self.x
        sig = self._sigmoid((self.x - self.theta) * 4.0)
        x_new = -self.x + self.k * sig + drive
        spiked = 1.0 if self.x >= self.theta_spike else 0.0
        theta_new = self.beta * self.theta + self.gamma * spiked

        if not math.isfinite(x_new) or not math.isfinite(theta_new):
            raise FloatingPointError("Kilinc-Bhatt candidate state became non-finite")

        self.x = max(-5.0, min(5.0, x_new))
        self.theta = max(-5.0, min(5.0, theta_new))

        return 1 if self.x >= self.x_threshold and x_prev < self.x_threshold else 0

    def reset(self) -> None:
        self.x = 0.0
        self.theta = 0.0
