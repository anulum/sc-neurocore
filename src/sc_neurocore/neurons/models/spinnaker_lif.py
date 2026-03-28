# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SpiNNaker LIF — ARM Cortex-M4 digital. Furber 2014

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpiNNakerLIFNeuron:
    """SpiNNaker LIF — ARM Cortex-M4 digital. Furber 2014."""

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    i_offset: float = 0.0
    tau_refrac: float = 2.0
    refrac_count: float = 0.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        if self.refrac_count > 0:
            self.refrac_count -= self.dt
            return 0
        self.v += (-(self.v - self.v_rest) + (current + self.i_offset)) / self.tau_m * self.dt
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.refrac_count = self.tau_refrac
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.refrac_count = 0.0


# ── SPECIALIZED / MODERN ──────────────────────────────────────────
