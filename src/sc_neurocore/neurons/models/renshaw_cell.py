# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Renshaw Cell (Spinal Inhibitory Interneuron)

from __future__ import annotations

import math
from dataclasses import dataclass


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


@dataclass
class RenshawCell:
    """Renshaw cell — spinal inhibitory interneuron for recurrent inhibition.

    WB gating core with strong adaptation to produce burst-then-decay
    response to motor axon collateral input.

    Reference: Renshaw (1941); Windhorst (1996) Prog Neurobiol 46(5).
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    adapt: float = 0.0
    g_na: float = 35.0
    g_k: float = 9.0
    g_adapt: float = 5.0
    g_l: float = 0.12
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    tau_adapt: float = 50.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def step(self, current: float = 0.0) -> int:
        v_prev = self.v
        n_sub = max(1, int(0.5 / max(self.dt, 0.001)))
        for _ in range(n_sub):
            am = _safe_rate(0.1, 35.0, self.v, 10.0, 1.0)
            bm = 4.0 * math.exp(-(self.v + 60.0) / 18.0)
            m_inf = am / (am + bm)
            ah = 0.07 * math.exp(-(self.v + 58.0) / 20.0)
            bh = 1.0 / (1.0 + math.exp(-(self.v + 28.0) / 10.0))
            an = _safe_rate(0.01, 34.0, self.v, 10.0, 0.1)
            bn = 0.125 * math.exp(-(self.v + 44.0) / 80.0)

            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt

            adapt_inf = 1.0 / (1.0 + math.exp(-(self.v + 30.0) / 5.0))
            self.adapt += (adapt_inf - self.adapt) / self.tau_adapt * self.dt

            i_na = self.g_na * m_inf**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_adapt = self.g_adapt * self.adapt * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_adapt - i_l + current) / self.c_m * self.dt

        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.8
        self.n = 0.1
        self.adapt = 0.0
