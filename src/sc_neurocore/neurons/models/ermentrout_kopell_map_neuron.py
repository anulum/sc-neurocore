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

    def step(self, current: float = 0.0) -> int:
        inp = self.gain * current
        theta_prev = self.theta

        d_theta = (1.0 - math.cos(self.theta)) + (1.0 + math.cos(self.theta)) * inp
        self.theta += self.dt * d_theta

        fired = 1 if self.theta >= self.theta_threshold and theta_prev < self.theta_threshold else 0

        two_pi = 2.0 * math.pi
        if self.theta >= two_pi:
            self.theta -= two_pi
        if self.theta < 0.0:
            self.theta += two_pi

        if not math.isfinite(self.theta):
            self.theta = 0.0

        return fired

    def reset(self) -> None:
        self.theta = 0.0
