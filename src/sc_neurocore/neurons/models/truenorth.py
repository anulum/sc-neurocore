# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Merolla 2014 — IBM TrueNorth digital neuron

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrueNorthNeuron:
    """Merolla 2014 — IBM TrueNorth digital neuron.

    Reference: Merolla, P.A. et al. (2014). Science 345:668–673.
    """

    v: int = 0
    leak: int = 0
    threshold: int = 100
    v_reset: int = 0

    def step(self, weighted_input: int) -> int:
        self.v = self.v + weighted_input - self.leak
        if self.v >= self.threshold:
            self.v = self.v_reset
            return 1
        if self.v < -self.threshold:
            self.v = self.v_reset
        return 0

    def reset(self) -> None:
        self.v = 0
