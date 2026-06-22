# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — KLIF — LIF with learnable scaling factor k

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KLIFNeuron:
    """KLIF — LIF with learnable scaling factor k.

    V[t+1] = alpha * V[t] + k * I; spike when V >= threshold.
    The scaling factor k is a trainable parameter for SNN backprop.

    Reference: Jiang, C. & Zhang, Y. (2024). Neural Comput. 36(8):1546–1564.
    """

    v: float = 0.0
    k: float = 1.0  # learnable scaling factor
    tau: float = 10.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dt: float = 1.0
    alpha: float = field(init=False)

    def __post_init__(self) -> None:
        self.alpha = np.exp(-self.dt / self.tau)

    def step(self, current: float) -> int:
        self.v = self.alpha * self.v + self.k * current
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = 0.0
