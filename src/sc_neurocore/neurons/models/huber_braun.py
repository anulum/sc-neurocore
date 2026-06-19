# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Braun, Huber et al. 1998 — cold receptor,

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HuberBraunNeuron:
    """Braun, Huber et al. 1998 — cold receptor, temperature-dependent.

    4 ODEs: V, a_sd (slow depolarizing), a_sr (slow repolarizing), a_r.

    Reference: Braun, H.A. et al. (1998). Int. J. Bifurcation Chaos 8:881–889.
    """

    v: float = -50.0
    a_sd: float = 0.0
    a_sr: float = 0.0
    g_sd: float = 1.5
    g_sr: float = 0.4
    g_l: float = 0.1
    e_sd: float = 50.0
    e_sr: float = -90.0
    e_l: float = -60.0
    tau_sd: float = 10.0
    tau_sr: float = 20.0
    eta: float = 0.012
    dt: float = 0.1
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        sd_inf = 1.0 / (1.0 + np.exp(-(self.v + 40.0) / 6.0))
        sr_inf = 1.0 / (1.0 + np.exp((self.v + 40.0) / 6.0))
        self.a_sd += (sd_inf - self.a_sd) / self.tau_sd * self.dt
        self.a_sr += (sr_inf - self.a_sr) / self.tau_sr * self.dt
        i_sd = self.g_sd * self.a_sd * (self.v - self.e_sd)
        i_sr = self.g_sr * self.a_sr * (self.v - self.e_sr)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_sd - i_sr - i_l + current + self.eta * np.random.randn()) * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -50.0
        self.a_sd, self.a_sr = 0.0, 0.0
