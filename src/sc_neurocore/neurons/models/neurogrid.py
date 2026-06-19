# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Boahen 2014 — Neurogrid subthreshold analog 2-compartment

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NeuroGridNeuron:
    """Boahen 2014 — Neurogrid subthreshold analog 2-compartment.

    Soma (v_s) and dendrite (v_d) coupled by conductance g_c.
    Soma is LIF with exponential spike initiation.
    Dendrite integrates synaptic input.

    Reference: Benjamin, B.V. et al. (2014). Proc. IEEE 102:699–716.
    """

    v_s: float = -65.0
    v_d: float = -65.0
    tau_s: float = 20.0  # ms, soma
    tau_d: float = 50.0  # ms, dendrite
    g_c: float = 0.5  # inter-compartment coupling
    delta_t: float = 2.0  # mV, exponential slope
    v_rest: float = -65.0
    v_threshold: float = -50.0
    v_peak: float = 20.0
    v_reset: float = -65.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        # Dendrite integrates input
        dv_d = (-(self.v_d - self.v_rest) + current - self.g_c * (self.v_d - self.v_s)) / self.tau_d
        self.v_d += dv_d * self.dt
        # Soma: EIF-like with dendritic coupling
        exp_term = self.delta_t * np.exp(min((self.v_s - self.v_threshold) / self.delta_t, 20.0))
        dv_s = (
            -(self.v_s - self.v_rest) + exp_term + self.g_c * (self.v_d - self.v_s)
        ) / self.tau_s
        self.v_s += dv_s * self.dt
        if self.v_s >= self.v_peak:
            self.v_s = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v_s, self.v_d = -65.0, -65.0
