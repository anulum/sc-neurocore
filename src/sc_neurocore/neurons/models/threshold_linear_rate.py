# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Threshold-linear (ReLU) rate neuron. Dayan & Abbott 2001

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThresholdLinearRateNeuron:
    """Threshold-linear (ReLU) rate neuron. Dayan & Abbott 2001.

    r = gain * max(0, input - theta)
    """

    r: float = 0.0
    theta: float = 0.0
    gain: float = 1.0

    def step(self, current: float) -> float:
        self.r = self.gain * max(0.0, current - self.theta)
        return self.r

    def reset(self):
        self.r = 0.0
