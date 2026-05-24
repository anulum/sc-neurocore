# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complementary LIF — ICML 2024, dual positive/negative

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class ComplementaryLIFNeuron:
    """Complementary LIF — ICML 2024, dual positive/negative spike paths.

    Maintains separate excitatory/inhibitory membrane potentials; spike
    emitted when their difference exceeds threshold.

    Reference: Hunsberger, E. & Eliasmith, C. (2015). Neural Comput. 27(12):2548–2586.
    """

    v_pos: float = 0.0
    v_neg: float = 0.0
    tau: float = 10.0
    v_threshold: float = 1.0
    dt: float = 1.0
    alpha: float = field(init=False)

    def __post_init__(self) -> None:
        for field_name in ("v_pos", "v_neg"):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        for field_name in ("tau", "dt", "v_threshold"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        self.alpha = np.exp(-self.dt / self.tau)

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        inp_pos = max(current, 0.0)
        inp_neg = max(-current, 0.0)
        self.v_pos = self.alpha * self.v_pos + inp_pos
        self.v_neg = self.alpha * self.v_neg + inp_neg
        diff = self.v_pos - self.v_neg
        if diff >= self.v_threshold:
            self.v_pos = 0.0
            self.v_neg = 0.0
            return 1
        if diff <= -self.v_threshold:
            self.v_pos = 0.0
            self.v_neg = 0.0
            return -1
        return 0

    def reset(self) -> None:
        self.v_pos, self.v_neg = 0.0, 0.0
