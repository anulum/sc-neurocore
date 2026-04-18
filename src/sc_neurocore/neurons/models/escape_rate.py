# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gerstner 2000 — stochastic threshold (escape noise model)

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sc_neurocore.utils.numerics import safe_exp


@dataclass
class EscapeRateNeuron:
    """Gerstner 2000 — stochastic threshold (escape noise model)."""

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    rho_0: float = 0.001
    delta_u: float = 3.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.v += (-(self.v - self.v_rest) + self.resistance * current) / self.tau_m * self.dt
        rate = self.rho_0 * safe_exp((self.v - self.v_threshold) / self.delta_u)
        p_spike = rate * self.dt
        if np.random.random() < p_spike:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
