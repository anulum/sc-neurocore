# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benda & Herz 2003 — Spike Frequency Adaptation IF

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class SFANeuron:
    """Benda & Herz 2003 — Spike Frequency Adaptation IF.

    Reference: Benda, J. & Herz, A.V.M. (2003). Neural Comput. 15:2523–2564.
    """

    v: float = -70.0
    g_sfa: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    tau_sfa: float = 200.0
    delta_g: float = 0.5
    e_k: float = -80.0
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        for field in ("v", "v_rest", "v_reset", "v_threshold", "e_k"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if not math.isfinite(self.g_sfa) or self.g_sfa < 0.0:
            raise ValueError("g_sfa must be finite and non-negative")
        for field in ("tau_m", "tau_sfa", "resistance", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if not math.isfinite(self.delta_g) or self.delta_g < 0.0:
            raise ValueError("delta_g must be finite and non-negative")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        self.v += (
            (-(self.v - self.v_rest) - self.g_sfa * (self.v - self.e_k) + self.resistance * current)
            / self.tau_m
            * self.dt
        )
        self.g_sfa *= np.exp(-self.dt / self.tau_sfa)
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.g_sfa += self.delta_g
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.g_sfa = 0.0
