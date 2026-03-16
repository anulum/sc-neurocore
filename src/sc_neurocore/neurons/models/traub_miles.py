# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TraubMilesNeuron:
    """Traub & Miles 1991 — reduced hippocampal CA3 pyramidal."""

    v: float = -67.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.3
    g_na: float = 100.0
    g_k: float = 80.0
    g_l: float = 0.1
    e_na: float = 50.0
    e_k: float = -100.0
    e_l: float = -67.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(10):
            d = self.v + 54.0
            am = 0.32 * d / (1.0 - np.exp(-d / 4.0)) if abs(d) > 1e-6 else 8.0
            d2 = self.v + 27.0
            bm = 0.28 * d2 / (np.exp(d2 / 5.0) - 1.0) if abs(d2) > 1e-6 else 5.6
            ah = 0.128 * np.exp(-(self.v + 50.0) / 18.0)
            bh = 4.0 / (1.0 + np.exp(-(self.v + 27.0) / 5.0))
            d3 = self.v + 52.0
            an = 0.032 * d3 / (1.0 - np.exp(-d3 / 5.0)) if abs(d3) > 1e-6 else 0.32
            bn = 0.5 * np.exp(-(self.v + 57.0) / 40.0)
            self.m += (am * (1 - self.m) - bm * self.m) * self.dt
            self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
            self.n += (an * (1 - self.n) - bn * self.n) * self.dt
            i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)
            self.v += (-i_na - i_k - i_l + current) * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.m, self.h, self.n = -67.0, 0.05, 0.6, 0.3
