# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic Integrate-and-Fire — canonical Type-I excitability

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class QuadraticIFNeuron:
    """Quadratic Integrate-and-Fire — canonical Type-I excitability.

    dv/dt = v² + I
    Reset when v >= v_peak.

    Reference: Latham, P.E. et al. (2000). J. Neurophysiol. 83:808–827.
    """

    v: float = -1.0
    v_reset: float = -1.0
    v_peak: float = 1.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        for field in ("v", "v_reset", "v_peak"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v_reset >= self.v_peak:
            raise ValueError("v_peak must be greater than v_reset")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self.v += (self.v**2 + current) * self.dt
        if self.v >= self.v_peak:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_reset
