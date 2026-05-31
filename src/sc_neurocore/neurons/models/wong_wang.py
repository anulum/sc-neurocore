# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wong & Wang 2006 — reduced decision-making attractor model

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class WongWangUnit:
    """Wong & Wang 2006 — reduced decision-making attractor model.

    Reference: Wong, K.-F. & Wang, X.-J. (2006). J. Neurosci. 26:1314–1328.
    """

    s1: float = 0.1
    s2: float = 0.1
    tau_s: float = 0.1
    gamma: float = 0.641
    j_n: float = 0.2609
    j_cross: float = 0.0497
    i_0: float = 0.3255
    sigma: float = 0.02
    dt: float = 0.001

    def __post_init__(self) -> None:
        for name in (
            "s1",
            "s2",
            "tau_s",
            "gamma",
            "j_n",
            "j_cross",
            "i_0",
            "sigma",
            "dt",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in ("tau_s", "gamma", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("j_n", "j_cross", "sigma"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self._validate_state(self.s1, self.s2)

    @staticmethod
    def _validate_state(s1: float, s2: float) -> tuple[float, float]:
        state1 = float(s1)
        state2 = float(s2)
        if (
            not math.isfinite(state1)
            or not math.isfinite(state2)
            or state1 < 0.0
            or state1 > 1.0
            or state2 < 0.0
            or state2 > 1.0
        ):
            raise FloatingPointError("Wong-Wang gating state must remain in [0, 1]")
        return state1, state2

    def _phi(self, i_syn: float) -> float:
        i_value = float(i_syn)
        if not math.isfinite(i_value):
            raise ValueError("synaptic current must be finite")
        a, b, d = 270.0, 108.0, 0.154
        x = a * i_value - b
        if abs(x) < 1e-6:
            return 1.0 / d
        exponent = -d * x
        if exponent > 700.0:
            return 0.0
        response = x / (1.0 - math.exp(exponent))
        if not math.isfinite(response) or response < 0.0:
            raise FloatingPointError("Wong-Wang transfer response must be finite")
        return response

    def step(self, stim1: float = 0.0, stim2: float = 0.0) -> tuple[float, float]:
        # `np.clip(scalar, 0, 1)` builds a numpy generic wrapper for every
        # call — measured as 45 % of step() on cProfile. Replace with
        # built-in branch; preserves semantics, gives ~2× throughput.
        drive1 = float(stim1)
        drive2 = float(stim2)
        if not math.isfinite(drive1) or not math.isfinite(drive2):
            raise ValueError("stimuli must be finite")

        s1, s2 = self._validate_state(self.s1, self.s2)
        noise1 = self.sigma * float(np.random.randn())
        noise2 = self.sigma * float(np.random.randn())
        if not math.isfinite(noise1) or not math.isfinite(noise2):
            raise FloatingPointError("Wong-Wang noise sample became non-finite")
        i1 = self.j_n * s1 - self.j_cross * s2 + self.i_0 + drive1 + noise1
        i2 = self.j_n * s2 - self.j_cross * s1 + self.i_0 + drive2 + noise2
        r1, r2 = self._phi(i1), self._phi(i2)
        next_s1 = s1 + (-s1 / self.tau_s + (1.0 - s1) * self.gamma * r1) * self.dt
        next_s2 = s2 + (-s2 / self.tau_s + (1.0 - s2) * self.gamma * r2) * self.dt
        if not math.isfinite(next_s1) or not math.isfinite(next_s2):
            raise FloatingPointError("Wong-Wang candidate state became non-finite")
        self.s1 = min(1.0, max(0.0, next_s1))
        self.s2 = min(1.0, max(0.0, next_s2))
        return (r1, r2)

    def reset(self) -> None:
        self.s1, self.s2 = 0.1, 0.1
