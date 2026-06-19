# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Montbrio, Pazo & Roxin 2015 — exact mean-field of

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErmentroutKopellPopulation:
    """Montbrio, Pazo & Roxin 2015 — exact mean-field of QIF/theta network.

    Reference: Ermentrout, G.B. & Kopell, N. (1986). SIAM J. Appl. Math. 46:233–253.
    """

    r: float = 0.1
    v: float = -2.0
    tau: float = 1.0
    delta: float = 1.0
    eta_bar: float = -5.0
    j: float = 15.0
    dt: float = 0.01

    def step(self, ext_input: float = 0.0) -> float:
        dr = (self.delta / (np.pi * self.tau) + 2.0 * self.r * self.v) / self.tau * self.dt
        dv = (
            (
                self.v**2
                + self.eta_bar
                + ext_input
                + self.j * self.tau * self.r
                - (np.pi * self.tau * self.r) ** 2
            )
            / self.tau
            * self.dt
        )
        self.r = max(0.0, self.r + dr)
        self.v += dv
        return self.r

    def reset(self) -> None:
        self.r, self.v = 0.1, -2.0


# ── MAP-BASED ──────────────────────────────────────────────────────
