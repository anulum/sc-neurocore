# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Non-leaky integrate-and-fire. Lapicque 1907 (no leak)

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerfectIntegratorNeuron:
    """Non-leaky integrate-and-fire. Lapicque 1907 (no leak).

    dV/dt = I / C
    """

    v: float = 0.0
    c_m: float = 1.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        self.v += current / self.c_m * self.dt
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_reset
