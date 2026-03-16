# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Indiveri et al. 2011 — DYNAP-SE differential-pair integrator

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DPINeuron:
    """Indiveri et al. 2011 — DYNAP-SE differential-pair integrator.

    Subthreshold log-domain dynamics modelling analog VLSI circuits.
    tau dI_mem/dt = -I_mem + I_syn + I_leak
    Spike when I_mem >= I_threshold, reset to I_reset.
    All variables in current domain (nA), mirroring transistor currents.
    """

    i_mem: float = 0.0
    i_threshold: float = 1.0
    i_reset: float = 0.0
    i_leak: float = 0.01
    tau: float = 20.0
    gain: float = 1.0
    dt: float = 1.0

    def step(self, i_syn: float) -> int:
        di = (-self.i_mem + self.gain * i_syn + self.i_leak) / self.tau * self.dt
        self.i_mem += di
        self.i_mem = max(self.i_mem, 0.0)
        if self.i_mem >= self.i_threshold:
            self.i_mem = self.i_reset
            return 1
        return 0

    def reset(self):
        self.i_mem = 0.0
