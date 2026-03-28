# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Loihi CUBA LIF — Intel Loihi fixed-point neuron. Davies 2018

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoihiCUBANeuron:
    """Loihi CUBA LIF — Intel Loihi fixed-point neuron. Davies 2018."""

    v: int = 0
    u: int = 0
    tau_v: int = 10
    tau_u: int = 5
    v_threshold: int = 1000
    v_reset: int = 0

    def step(self, weighted_input: int) -> int:
        self.u = self.u - self.u // self.tau_u + weighted_input
        self.v = self.v - self.v // self.tau_v + self.u
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v, self.u = 0, 0
