# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ermentrout-Kopell Canonical Type I Map Neuron

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ErmentroutKopellMapNeuron:
    """Ermentrout-Kopell 1986 canonical Type I (theta neuron) map.

    The canonical model for Type I (saddle-node) excitability. Phase
    variable θ advances on a circle; spike occurs when θ crosses π.

    θ(n+1) = θ(n) + dt · [(1 - cos θ) + (1 + cos θ) · I]

    Reference: Ermentrout & Kopell (1986) SIAM J Appl Math 46:233–253.
    """

    theta: float = 0.0
    dt: float = 0.1
    gain: float = 1.0
    theta_threshold: float = math.pi

    def __post_init__(self) -> None:
        for name in ("theta", "dt", "gain", "theta_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    @staticmethod
    def _validate_theta(theta: float) -> float:
        value = float(theta)
        if not math.isfinite(value):
            raise FloatingPointError("Ermentrout-Kopell phase state must be finite")
        return value

    def step(self, current: float = 0.0) -> int:
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")

        theta = self._validate_theta(self.theta)
        inp = self.gain * drive
        if not math.isfinite(inp):
            raise FloatingPointError("Ermentrout-Kopell input drive became non-finite")
        theta_prev = theta

        cos_theta = math.cos(theta)
        d_theta = (1.0 - cos_theta) + (1.0 + cos_theta) * inp
        theta_next = theta + self.dt * d_theta
        if not math.isfinite(d_theta) or not math.isfinite(theta_next):
            raise FloatingPointError("Ermentrout-Kopell candidate phase became non-finite")

        fired = 1 if theta_next >= self.theta_threshold and theta_prev < self.theta_threshold else 0
        two_pi = 2.0 * math.pi
        self.theta = theta_next % two_pi

        return fired

    def reset(self) -> None:
        self.theta = 0.0
