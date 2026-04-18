# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque 1907 — classical RC integrate-and-fire

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LapicqueNeuron:
    """Lapicque 1907 — classical RC integrate-and-fire.

    tau * dv/dt = -(v - v_rest) + R * I
    """

    v: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau: float = 20.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        dv = (-(self.v - self.v_rest) + self.resistance * current) / self.tau * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
