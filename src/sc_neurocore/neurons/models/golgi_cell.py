# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cerebellar Golgi Cell (Solinas 2007)

from __future__ import annotations

import math
from dataclasses import dataclass


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


def _boltz(v: float, vh: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-(v - vh) / k))


@dataclass
class GolgiCell:
    """Cerebellar Golgi cell — Solinas et al. 2007 full model.

    11 ionic currents: INa_t (m³h), INa_p, IKdr (n⁴), IKA (a³b),
    IKM (w), ICaT (mT²s), ICaN (c²), IBK (V+Ca²⁺), ISK (Ca²⁺),
    Ih (r), IL. Spontaneously active 3–10 Hz.

    Reference: Solinas et al. (2007) Front Cell Neurosci 1:2.
    """

    v: float = -60.0
    m: float = 0.02
    h: float = 0.85
    p_na: float = 0.01
    n: float = 0.05
    a: float = 0.1
    b: float = 0.8
    w: float = 0.01
    m_t: float = 0.01
    s: float = 0.9
    c_n: float = 0.01
    r: float = 0.1
    ca: float = 0.05
    g_na_t: float = 48.0
    g_na_p: float = 0.2
    g_kdr: float = 16.0
    g_ka: float = 8.0
    g_km: float = 1.0
    g_cat: float = 0.5
    g_can: float = 1.0
    g_bk: float = 3.0
    g_sk: float = 1.0
    g_h: float = 0.1
    g_l: float = 0.05
    e_na: float = 55.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_h: float = -40.0
    e_l: float = -55.0
    c_m: float = 1.0
    tau_ca: float = 200.0
    kd_bk: float = 1.0
    kd_sk: float = 0.5
    dt: float = 0.5
    sub_steps: int = 10
    gain: float = 1.0

    def step(self, current: float = 0.0) -> int:
        inp = self.gain * current
        dt_sub = self.dt / self.sub_steps
        v_prev = self.v

        for _ in range(self.sub_steps):
            v = self.v

            alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            beta_m = 4.0 * math.exp(-(v + 60.0) / 18.0)
            alpha_h = 0.07 * math.exp(-(v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))
            self.m += dt_sub * 5.0 * (alpha_m * (1.0 - self.m) - beta_m * self.m)
            self.h += dt_sub * 5.0 * (alpha_h * (1.0 - self.h) - beta_h * self.h)

            pna_inf = _boltz(v, -48.0, 5.0)
            tau_pna = 5.0 + 20.0 / max(0.01, 1.0 + ((v + 48.0) / 10.0) ** 2)
            self.p_na += dt_sub * (pna_inf - self.p_na) / tau_pna

            alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            beta_n = 0.125 * math.exp(-(v + 44.0) / 80.0)
            self.n += dt_sub * 5.0 * (alpha_n * (1.0 - self.n) - beta_n * self.n)

            a_inf = _boltz(v, -27.0, 16.0)
            self.a += dt_sub * (a_inf - self.a) / 2.0
            b_inf = _boltz(v, -80.0, -6.0)
            self.b += dt_sub * (b_inf - self.b) / 15.0

            w_inf = _boltz(v, -35.0, 10.0)
            tau_w = 100.0 / (3.3 * math.exp((v + 35.0) / 20.0) + math.exp(-(v + 35.0) / 20.0))
            self.w += dt_sub * (w_inf - self.w) / tau_w

            mt_inf = _boltz(v, -52.0, 5.0)
            self.m_t += dt_sub * (mt_inf - self.m_t) / 1.0
            s_inf = _boltz(v, -60.0, -6.5)
            tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0) ** 2)
            self.s += dt_sub * (s_inf - self.s) / tau_s

            cn_inf = _boltz(v, -20.0, 5.0)
            tau_cn = 2.0 + 10.0 / max(0.01, 1.0 + ((v + 20.0) / 10.0) ** 2)
            self.c_n += dt_sub * (cn_inf - self.c_n) / tau_cn

            r_inf = _boltz(v, -80.0, -10.0)
            tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0) ** 2)
            self.r += dt_sub * (r_inf - self.r) / tau_r

            for attr in ('m', 'h', 'p_na', 'n', 'a', 'b', 'w', 'm_t', 's', 'c_n', 'r'):
                setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))

            i_cat = self.g_cat * self.m_t ** 2 * self.s * (v - self.e_ca)
            i_can = self.g_can * self.c_n ** 2 * (v - self.e_ca)
            ca_entry = -(i_cat + i_can) * 0.001 if (i_cat + i_can) < 0.0 else 0.0
            self.ca += dt_sub * (ca_entry - self.ca / self.tau_ca)
            self.ca = max(0.0, self.ca)

            ca2 = self.ca ** 2
            kd2 = self.kd_bk ** 2
            bk_v = _boltz(v, 100.0 - 120.0 * ca2 / (ca2 + kd2), 15.0)
            sk_inf = ca2 / (ca2 + self.kd_sk ** 2)

            i_na_t = self.g_na_t * self.m ** 3 * self.h * (v - self.e_na)
            i_na_p = self.g_na_p * self.p_na * (v - self.e_na)
            i_kdr = self.g_kdr * self.n ** 4 * (v - self.e_k)
            i_ka = self.g_ka * self.a ** 3 * self.b * (v - self.e_k)
            i_km = self.g_km * self.w * (v - self.e_k)
            i_bk = self.g_bk * bk_v * (v - self.e_k)
            i_sk = self.g_sk * sk_inf * (v - self.e_k)
            i_h = self.g_h * self.r * (v - self.e_h)
            i_l = self.g_l * (v - self.e_l)

            dv_val = (-(i_na_t + i_na_p + i_kdr + i_ka + i_km + i_cat + i_can
                        + i_bk + i_sk + i_h + i_l) + inp) / self.c_m
            self.v += dt_sub * dv_val

        self.v = max(-100.0, min(60.0, self.v))
        if not math.isfinite(self.v):
            self.v = -60.0
        if not math.isfinite(self.ca):
            self.ca = 0.05

        return 1 if self.v >= 0.0 and v_prev < 0.0 else 0

    def reset(self) -> None:
        self.v = -60.0
        self.m = 0.02
        self.h = 0.85
        self.p_na = 0.01
        self.n = 0.05
        self.a = 0.1
        self.b = 0.8
        self.w = 0.01
        self.m_t = 0.01
        self.s = 0.9
        self.c_n = 0.01
        self.r = 0.1
        self.ca = 0.05
