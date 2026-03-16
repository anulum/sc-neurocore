# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Loihi2Neuron:
    """Intel Loihi 2, 2021 — programmable 3-state-variable neuron.

    State variables (s1, s2, s3) with configurable decay, threshold,
    and cross-coupling. Generalises CUBA, COBA, and Izhikevich on-chip.
    All integer arithmetic with configurable bit-shift decays.
    """

    s1: int = 0  # primary state (membrane potential analogue)
    s2: int = 0  # secondary state (synaptic current / adaptation)
    s3: int = 0  # tertiary state (slow modulation)
    tau1: int = 10  # s1 decay divisor
    tau2: int = 5  # s2 decay divisor
    tau3: int = 50  # s3 decay divisor
    w12: int = 1  # s2 → s1 coupling weight
    w13: int = 0  # s3 → s1 coupling weight
    w23: int = 0  # s3 → s2 coupling weight
    s1_threshold: int = 1000
    s1_reset: int = 0
    s3_incr: int = 10  # s3 increment on spike (adaptation)

    def step(self, weighted_input: int) -> int:
        self.s3 -= self.s3 // self.tau3
        self.s2 = self.s2 - self.s2 // self.tau2 + weighted_input + self.w23 * self.s3
        self.s1 = self.s1 - self.s1 // self.tau1 + self.w12 * self.s2 + self.w13 * self.s3
        if self.s1 >= self.s1_threshold:
            self.s1 = self.s1_reset
            self.s3 += self.s3_incr
            return 1
        return 0

    def reset(self):
        self.s1, self.s2, self.s3 = 0, 0, 0
