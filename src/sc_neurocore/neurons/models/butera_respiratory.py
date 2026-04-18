# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Butera, Rinzel & Smith 1999 — pre-Botzinger respiratory

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ButeraRespiratoryNeuron:
    """Butera, Rinzel & Smith 1999 — pre-Botzinger respiratory neuron.

    Reference: Butera, R.J. et al. (1999). J. Neurophysiol. 82:382–397.
    """

    v: float = -50.0
    n: float = 0.01
    h_nap: float = 0.5
    g_na: float = 28.0
    g_nap: float = 2.8
    g_k: float = 11.2
    g_l: float = 2.8
    e_na: float = 50.0
    e_k: float = -85.0
    e_l: float = -65.0
    e_syn: float = -10.0
    tau_h: float = 10000.0
    dt: float = 0.1
    v_threshold: float = -20.0

    @staticmethod
    def _sexp(x: float) -> float:
        return float(np.exp(np.clip(x, -500, 500)))

    @staticmethod
    def _scosh(x: float) -> float:
        cx = np.clip(x, -500, 500)
        return float(np.cosh(cx))

    def step(self, current: float) -> int:
        v_prev = self.v
        m_na_inf = 1.0 / (1.0 + self._sexp(-(self.v + 34.0) / 5.0))
        m_nap_inf = 1.0 / (1.0 + self._sexp(-(self.v + 40.0) / 6.0))
        h_nap_inf = 1.0 / (1.0 + self._sexp((self.v + 48.0) / 6.0))
        n_inf = 1.0 / (1.0 + self._sexp(-(self.v + 29.0) / 4.0))
        tau_n = 10.0 / max(self._scosh((self.v + 29.0) / 8.0), 1e-12)
        tau_h = self.tau_h / max(self._scosh((self.v + 48.0) / 12.0), 1e-12)
        i_na = self.g_na * m_na_inf**3 * (1.0 - self.n) * (self.v - self.e_na)
        i_nap = self.g_nap * m_nap_inf * self.h_nap * (self.v - self.e_na)
        i_k = self.g_k * self.n**4 * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_na - i_nap - i_k - i_l + current) * self.dt
        self.v = float(np.clip(self.v, -200, 100))
        self.n += (n_inf - self.n) / max(tau_n, 0.01) * self.dt
        self.n = float(np.clip(self.n, 0, 1))
        self.h_nap += (h_nap_inf - self.h_nap) / max(tau_h, 0.1) * self.dt
        self.h_nap = float(np.clip(self.h_nap, 0, 1))
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v, self.n, self.h_nap = -50.0, 0.01, 0.5
