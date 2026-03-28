# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ExpIFNeuron:
    """Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003."""

    v: float = -65.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    v_rh: float = -55.0
    delta_t: float = 2.0
    tau: float = 20.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        exp_term = self.delta_t * np.exp(np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0))
        dv = (-(self.v - self.v_rest) + exp_term + current) / self.tau * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
