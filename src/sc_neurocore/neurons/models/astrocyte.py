# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Li & Bhatt 1994 — astrocyte Ca2+ signaling via IP3 receptor

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AstrocyteModel:
    """Li & Bhatt 1994 — astrocyte Ca2+ signaling via IP3 receptor.

    3 ODEs: Ca (cytosolic), h (IP3R de-inactivation), IP3.
    Ca release from ER through IP3 receptor + SERCA pump + leak.
    """

    ca: float = 0.05  # uM, cytosolic Ca
    h: float = 0.8  # IP3R de-inactivation gate
    ip3: float = 0.5  # uM
    v_er: float = 0.9  # uM/s, max ER release rate
    k_er: float = 0.15  # uM, SERCA pump half-activation
    v_serca: float = 0.4  # uM/s, max pump rate
    d1: float = 0.13  # uM, IP3 dissociation constant
    d2: float = 1.049  # uM, Ca inactivation dissociation
    d3: float = 0.9434  # uM, IP3 binding with Ca
    d5: float = 0.08234  # uM, Ca activation dissociation
    a2: float = 0.2  # uM^-1 s^-1, Ca inactivation rate
    c1: float = 0.185  # ER/cyt volume ratio
    leak: float = 0.01  # s^-1, ER leak rate
    ip3_prod: float = 0.0  # uM/s, external IP3 production
    ip3_decay: float = 0.14  # s^-1
    dt: float = 0.01  # s

    def step(self, current: float) -> float:
        """Return cytosolic Ca concentration (uM). current = glutamate-driven IP3 production."""
        # Li-Rinzel IP3R open probability
        m_inf = self.ip3 / (self.ip3 + self.d1)
        n_inf = self.ca / (self.ca + self.d5)
        j_channel = self.v_er * (m_inf * n_inf * self.h) ** 3 * (self.ca * self.c1 - self.ca)
        j_serca = self.v_serca * self.ca**2 / (self.ca**2 + self.k_er**2)
        j_leak = self.leak * (self.ca * self.c1 - self.ca)

        dca = j_channel - j_serca + j_leak
        q2 = self.d2 * (self.ip3 + self.d1) / (self.ip3 + self.d3)
        h_inf = q2 / (q2 + self.ca)
        tau_h = 1.0 / (self.a2 * (q2 + self.ca))
        dh = (h_inf - self.h) / max(tau_h, 1e-6)
        dip3 = current + self.ip3_prod - self.ip3_decay * self.ip3

        self.ca = max(0.0, self.ca + dca * self.dt)
        self.h = np.clip(self.h + dh * self.dt, 0.0, 1.0)
        self.ip3 = max(0.0, self.ip3 + dip3 * self.dt)
        return self.ca

    def reset(self):
        self.ca, self.h, self.ip3 = 0.05, 0.8, 0.5
