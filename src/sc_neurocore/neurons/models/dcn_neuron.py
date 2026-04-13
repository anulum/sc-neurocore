# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Deep Cerebellar Nuclei (DCN) Neuron

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


@dataclass
class DCNNeuron:
    """Deep cerebellar nuclei neuron — main output of the cerebellum.

    WB Na⁺/K⁺ core + T-type Ca²⁺ (rebound bursting), Ih (pacemaker),
    persistent Na⁺ (subthreshold), Ca²⁺-dependent AHP. 7 currents total.

    Reference: Llinás & Mühlethaler (1988) J Physiol 404:241;
    Jahnsen (1986) J Physiol 372:129.
    """

    v: float = -60.0
    h: float = 0.6
    n: float = 0.32
    p: float = 0.01
    s: float = 0.8
    r: float = 0.1
    ca: float = 0.05
    g_na: float = 35.0
    g_nap: float = 0.5
    g_k: float = 9.0
    g_t: float = 0.1
    g_ahp: float = 2.0
    g_h: float = 0.02
    g_l: float = 0.2
    e_na: float = 55.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_h: float = -40.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    tau_ca: float = 150.0
    kd_ahp: float = 0.5
    dt: float = 0.5
    v_threshold: float = -20.0
    gain: float = 1.0
    _sub_steps: int = field(default=20, repr=False)

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

            p_inf = 1.0 / (1.0 + math.exp(-(v + 48.0) / 5.0))
            tau_p = 5.0 + 15.0 / max(0.01, 1.0 + ((v + 48.0) / 10.0) ** 2)

            m_t_inf = 1.0 / (1.0 + math.exp(-(v + 52.0) / 5.0))
            s_inf = 1.0 / (1.0 + math.exp((v + 60.0) / 6.5))
            tau_s = 20.0 + 50.0 / (1.0 + math.exp((v + 65.0) / 10.0))

            r_inf = 1.0 / (1.0 + math.exp((v + 80.0) / 10.0))
            tau_r = 100.0 + 200.0 / (1.0 + math.exp((v + 70.0) / 10.0))

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h)
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n)
            self.p += sub_dt * (p_inf - self.p) / tau_p
            self.s += sub_dt * (s_inf - self.s) / tau_s
            self.r += sub_dt * (r_inf - self.r) / tau_r

            i_t = self.g_t * m_t_inf ** 2 * self.s * (v - self.e_ca)
            ca_entry = -i_t * 0.001 if i_t < 0.0 else 0.0
            self.ca += sub_dt * (ca_entry - self.ca / self.tau_ca)
            self.ca = max(0.0, self.ca)

            ahp_inf = self.ca ** 2 / (self.ca ** 2 + self.kd_ahp ** 2)

            i_na = self.g_na * m_inf ** 3 * self.h * (v - self.e_na)
            i_nap = self.g_nap * self.p * (v - self.e_na)
            i_k = self.g_k * self.n ** 4 * (v - self.e_k)
            i_ahp = self.g_ahp * ahp_inf * (v - self.e_k)
            i_h = self.g_h * self.r * (v - self.e_h)
            i_l = self.g_l * (v - self.e_l)

            dv_val = (-i_na - i_nap - i_k - i_t - i_ahp - i_h - i_l + inp) / self.c_m
            self.v += sub_dt * dv_val

            if self.v >= self.v_threshold:
                fired = 1
                self.v = -60.0
                self.ca += 0.2

        self.v = max(-100.0, min(60.0, self.v))
        if not math.isfinite(self.v):
            self.v = -60.0
            self.h = 0.6
            self.n = 0.32
        if not math.isfinite(self.ca):
            self.ca = 0.05
        self.h = max(0.0, min(1.0, self.h))
        self.n = max(0.0, min(1.0, self.n))
        self.p = max(0.0, min(1.0, self.p))
        self.s = max(0.0, min(1.0, self.s))
        self.r = max(0.0, min(1.0, self.r))
        return fired

    def reset(self) -> None:
        self.v = -60.0
        self.h = 0.6
        self.n = 0.32
        self.p = 0.01
        self.s = 0.8
        self.r = 0.1
        self.ca = 0.05
