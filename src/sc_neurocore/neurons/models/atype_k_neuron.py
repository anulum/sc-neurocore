# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A-type K⁺ (IA) Neuron

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


@dataclass
class ATypeKNeuron:
    """A-type K⁺ (IA) neuron — WB base + transient outward K⁺ current.

    IA activates rapidly at subthreshold voltages and inactivates over
    tens of milliseconds. Controls first-spike latency and interspike
    intervals.

    a∞ = 1 / (1 + exp(-(v + 50) / 20))   τ_a = 2 ms
    b∞ = 1 / (1 + exp((v + 70) / 6))     τ_b = 50 ms

    Reference: Connor & Stevens (1971) J Physiol 213:31;
    Hoffman et al. (1997) Nature 387:869.
    """

    v: float = -65.0
    h: float = 0.6
    n: float = 0.32
    a: float = 0.1
    b: float = 0.8
    g_na: float = 35.0
    g_k: float = 9.0
    g_a: float = 8.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
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

            a_inf = 1.0 / (1.0 + math.exp(-(v + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + math.exp((v + 70.0) / 6.0))

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h)
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n)
            self.a += sub_dt * (a_inf - self.a) / 2.0
            self.b += sub_dt * (b_inf - self.b) / 50.0

            i_na = self.g_na * m_inf**3 * self.h * (v - self.e_na)
            i_k = self.g_k * self.n**4 * (v - self.e_k)
            i_a = self.g_a * self.a**3 * self.b * (v - self.e_k)
            i_l = self.g_l * (v - self.e_l)

            dv = (-i_na - i_k - i_a - i_l + inp) / self.c_m
            self.v += sub_dt * dv

            if self.v >= self.v_threshold:
                fired = 1
                self.v = -65.0

        self.v = max(-100.0, min(60.0, self.v))
        if not math.isfinite(self.v):
            self.v = -65.0
            self.h = 0.6
            self.n = 0.32
        self.h = max(0.0, min(1.0, self.h))
        self.n = max(0.0, min(1.0, self.n))
        self.a = max(0.0, min(1.0, self.a))
        self.b = max(0.0, min(1.0, self.b))

        return fired

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.6
        self.n = 0.32
        self.a = 0.1
        self.b = 0.8
