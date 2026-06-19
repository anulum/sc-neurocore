# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Threshold-linear (ReLU) rate neuron. Dayan & Abbott 2001

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ThresholdLinearRateNeuron:
    """Threshold-linear (ReLU) rate neuron. Dayan & Abbott 2001.

    r = gain * max(0, input - theta)

    Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §15.2.
    """

    r: float = 0.0
    theta: float = 0.0
    gain: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ValueError("r must be finite and non-negative")
        if not math.isfinite(self.theta):
            raise ValueError("theta must be finite")
        if not math.isfinite(self.gain) or self.gain < 0.0:
            raise ValueError("gain must be finite and non-negative")

    def step(self, current: float) -> float:
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ValueError("runtime rate state must be finite and non-negative")
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        try:
            next_r = self.gain * max(0.0, current - self.theta)
        except OverflowError as exc:
            raise ValueError("rate output must remain finite") from exc
        if not math.isfinite(next_r) or next_r < 0.0:
            raise ValueError("rate output must remain finite and non-negative")
        self.r = next_r
        return next_r

    def reset(self) -> None:
        self.r = 0.0
