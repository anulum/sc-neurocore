# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta neuron — canonical Type-I on the unit circle

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ThetaNeuron:
    """Theta neuron — canonical Type-I on the unit circle.

    dθ/dt = (1 - cos θ) + (1 + cos θ) · I
    Spike when θ crosses π.
    Ermentrout & Kopell 1986.

    Reference: Ermentrout, G.B. & Kopell, N. (1986). SIAM J. Appl. Math. 46:233–253.
    """

    theta: float = 0.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(self.theta):
            raise ValueError("theta must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.theta = self._wrap_phase(self.theta)

    @staticmethod
    def _wrap_phase(theta: float) -> float:
        return ((theta + math.pi) % (2.0 * math.pi)) - math.pi

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        theta_prev = self.theta
        cos_theta = math.cos(self.theta)
        dtheta = ((1.0 - cos_theta) + (1.0 + cos_theta) * current) * self.dt
        next_theta = self.theta + dtheta
        if not math.isfinite(dtheta) or not math.isfinite(next_theta):
            raise ValueError("phase increment must be finite")
        spike = 1 if (theta_prev < math.pi * 0.99 and next_theta >= math.pi * 0.99) else 0
        self.theta = self._wrap_phase(next_theta)
        return spike

    def reset(self) -> None:
        self.theta = 0.0

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.theta):
            raise ValueError("runtime phase state must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("runtime dt must be finite and positive")
