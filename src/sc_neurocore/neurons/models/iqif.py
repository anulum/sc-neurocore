# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lo et al. 2021 — fixed-point quadratic integrate-and-fire

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntegerQIFNeuron:
    """Lo et al. 2021 — fixed-point quadratic integrate-and-fire.

    V[t+1] = V[t] + (V[t]^2 >> k) + I, all integer arithmetic.
    """

    v: int = 0
    k: int = 6  # right-shift for V^2
    v_threshold: int = 1024
    v_reset: int = -1024
    v_min: int = -2048

    def step(self, current: int) -> int:
        self.v = max(self.v_min, self.v + (self.v * self.v >> self.k) + current)
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = 0
