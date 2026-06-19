# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

    Reference: Postnov, D.E. et al. (2009). Neural Comput. 21:2746–2782.
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
    c0: float = 2.0  # uM, total cell calcium (conserved)
    c1: float = 0.185  # ER/cyt volume ratio
    leak: float = 0.01  # s^-1, ER leak rate
    ip3_prod: float = 0.0  # uM/s, external IP3 production
    ip3_decay: float = 0.14  # s^-1
    dt: float = 0.01  # s

    def __post_init__(self) -> None:
        if not np.isfinite(self.ca) or self.ca < 0.0:
            raise ValueError("ca must be finite and non-negative")
        if not np.isfinite(self.h) or not (0.0 <= self.h <= 1.0):
            raise ValueError("h must be finite and in [0, 1]")
        if not np.isfinite(self.ip3) or self.ip3 < 0.0:
            raise ValueError("ip3 must be finite and non-negative")
        for name in ("v_er", "k_er", "v_serca", "d1", "d2", "d3", "d5", "a2", "c0", "c1", "dt"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("leak", "ip3_prod", "ip3_decay"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.ca >= self.c0:
            raise ValueError("ca must be below total cell calcium c0")

    def step(self, current: float) -> float:
        """Return cytosolic Ca concentration (uM). current = glutamate-driven IP3 production."""
        if not np.isfinite(current) or current < 0.0:
            raise ValueError("current must be finite and non-negative")

        # Li-Rinzel IP3R open probability
        m_inf = self.ip3 / (self.ip3 + self.d1)
        n_inf = self.ca / (self.ca + self.d5)
        ca_er = (self.c0 - self.ca) / self.c1  # Li-Rinzel 1994 conservation
        j_channel = self.v_er * (m_inf * n_inf * self.h) ** 3 * (ca_er - self.ca)
        j_serca = self.v_serca * self.ca**2 / (self.ca**2 + self.k_er**2)
        j_leak = self.leak * (ca_er - self.ca)

        dca = j_channel - j_serca + j_leak
        q2 = self.d2 * (self.ip3 + self.d1) / (self.ip3 + self.d3)
        h_inf = q2 / (q2 + self.ca)
        tau_h = 1.0 / (self.a2 * (q2 + self.ca))
        dh = (h_inf - self.h) / max(tau_h, 1e-6)
        dip3 = current + self.ip3_prod - self.ip3_decay * self.ip3

        ca_next = self.ca + dca * self.dt
        if not np.isfinite(ca_next) or ca_next > self.c0:
            raise ValueError("calcium update must remain finite and within the total calcium pool")

        self.ca = max(0.0, ca_next)
        self.h = np.clip(self.h + dh * self.dt, 0.0, 1.0)
        self.ip3 = max(0.0, self.ip3 + dip3 * self.dt)
        return self.ca

    def reset(self) -> None:
        self.ca, self.h, self.ip3 = 0.05, 0.8, 0.5
