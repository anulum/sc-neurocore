# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — T-type Ca²⁺ (IT) Neuron

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


@dataclass
class TTypeCaNeuron:
    """T-type Ca²⁺ (IT) neuron — WB base + low-voltage-activated Ca²⁺ current.

    IT activates at subthreshold voltages (-65 to -50 mV) and inactivates
    slowly. When de-inactivated by hyperpolarisation, IT produces a
    low-threshold spike (LTS) that can trigger a burst of Na⁺ spikes.

    m_T∞ = 1 / (1 + exp(-(v + 52) / 5))
    s∞ = 1 / (1 + exp((v + 81) / 4))
    τ_s = 30 + 100 / (1 + exp((v + 75) / 10))

    Reference: Huguenard (1996) Annu Rev Physiol 58:329;
    Destexhe et al. (1996) J Neurophysiol 76:2049.
    """

    v: float = -65.0
    h: float = 0.6
    n: float = 0.32
    s: float = 0.9
    g_na: float = 35.0
    g_k: float = 9.0
    g_t: float = 0.1
    g_l: float = 0.2
    e_na: float = 55.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    dt: float = 0.5
    v_threshold: float = -20.0
    gain: float = 1.0
    _sub_steps: int = field(default=50, repr=False)

    def step(self, current: float = 0.0) -> int:
        inp = self.gain * current
        sub_dt = self.dt / self._sub_steps
        fired = 0

        for _ in range(self._sub_steps):
            v = self.v

            alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            beta_m = 4.0 * math.exp(-(v + 60.0) / 18.0)
            m_inf = alpha_m / (alpha_m + beta_m)

            alpha_h = 0.07 * math.exp(-(v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))

            alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            beta_n = 0.125 * math.exp(-(v + 44.0) / 80.0)

            m_t_inf = 1.0 / (1.0 + math.exp(-(v + 52.0) / 5.0))
            s_inf = 1.0 / (1.0 + math.exp((v + 81.0) / 4.0))
            tau_s = 30.0 + 100.0 / (1.0 + math.exp((v + 75.0) / 10.0))

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h)
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n)
            self.s += sub_dt * (s_inf - self.s) / tau_s

            i_na = self.g_na * m_inf**3 * self.h * (v - self.e_na)
            i_k = self.g_k * self.n**4 * (v - self.e_k)
            i_t = self.g_t * m_t_inf**2 * self.s * (v - self.e_ca)
            i_l = self.g_l * (v - self.e_l)

            dv = (-i_na - i_k - i_t - i_l + inp) / self.c_m
            self.v += sub_dt * dv

            if self.v >= self.v_threshold:
                fired = 1
                self.v = -65.0
                self.s *= 0.3

        self.v = max(-100.0, min(60.0, self.v))
        if not math.isfinite(self.v):
            self.v = -65.0
            self.h = 0.6
            self.n = 0.32
        self.h = max(0.0, min(1.0, self.h))
        self.n = max(0.0, min(1.0, self.n))
        self.s = max(0.0, min(1.0, self.s))

        return fired

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.6
        self.n = 0.32
        self.s = 0.9
