# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BrainScaleS-2 — analog AdEx (1000x real-time). Schemmel 2010

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BrainScaleSAdExNeuron:
    """BrainScaleS-2 — analog AdEx (1000x real-time). Schemmel 2010.

    Reference: Schemmel, J. et al. (2010). Proc. ISCAS 2010: 1947–1950.
    """

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    delta_t: float = 2.0
    v_rh: float = -55.0
    tau: float = 20.0
    tau_w: float = 100.0
    a: float = 0.5
    b: float = 7.0
    hw_speedup: float = 1000.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        dt_hw = self.dt * self.hw_speedup
        exp_arg = np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0)
        exp_term = self.delta_t * np.exp(exp_arg)
        dv = (
            (-(self.v - self.v_rest) + exp_term - self.w + current)
            / self.tau
            * (dt_hw / self.hw_speedup)
        )
        dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * (dt_hw / self.hw_speedup)
        self.v += dv
        self.w += dw
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.w += self.b
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.w = 0.0
