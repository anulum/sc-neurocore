# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — TU Dresden / SpiNNaker2 2024 — ARM Cortex-M4F software LIF

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpiNNaker2Neuron:
    """TU Dresden / SpiNNaker2 2024 — ARM Cortex-M4F software LIF.

    Fixed-point LIF on M4F with exponential decay via integer multiply-shift.
    Includes refractory counter and configurable precision.
    """

    v: int = 0
    v_rest: int = 0
    v_reset: int = 0
    v_threshold: int = 1024
    decay_mult: int = 243  # ~exp(-1/10) * 256, 8-bit fixed-point
    decay_shift: int = 8  # right-shift after multiply
    refrac_steps: int = 2
    _refrac_count: int = 0

    def step(self, current: int) -> int:
        if self._refrac_count > 0:
            self._refrac_count -= 1
            return 0
        self.v = (
            ((self.v - self.v_rest) * self.decay_mult >> self.decay_shift) + self.v_rest + current
        )
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self._refrac_count = self.refrac_steps
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self._refrac_count = 0
