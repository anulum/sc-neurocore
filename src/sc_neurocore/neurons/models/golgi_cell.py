# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
    x = (v - vh) / k
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _all_finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


def _voltage(value: float) -> bool:
    return math.isfinite(value) and -100.0 <= value <= 60.0


def _gate_alpha_beta(
    previous: float, alpha: float, beta: float, phi: float, dt: float
) -> float | None:
    total = phi * (alpha + beta)
    if not _all_finite(previous, alpha, beta, total, dt) or total <= 0.0:
        return None
    steady = alpha / (alpha + beta)
    return min(1.0, max(0.0, steady + (previous - steady) * math.exp(-total * dt)))


def _gate_inf(previous: float, steady: float, tau: float, dt: float) -> float | None:
    if not _all_finite(previous, steady, tau, dt) or tau <= 0.0:
        return None
    return min(1.0, max(0.0, steady + (previous - steady) * math.exp(-dt / tau)))


def _calcium_exact(previous: float, entry: float, tau: float, dt: float) -> float | None:
    if not _all_finite(previous, entry, tau, dt) or tau <= 0.0 or previous < 0.0:
        return None
    steady = entry * tau
    value = steady + (previous - steady) * math.exp(-dt / tau)
    if not math.isfinite(value):
        return None
    return max(0.0, value)


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

    def _valid_state(self) -> bool:
        gates = (
            self.m,
            self.h,
            self.p_na,
            self.n,
            self.a,
            self.b,
            self.w,
            self.m_t,
            self.s,
            self.c_n,
            self.r,
        )
        conductances = (
            self.g_na_t,
            self.g_na_p,
            self.g_kdr,
            self.g_ka,
            self.g_km,
            self.g_cat,
            self.g_can,
            self.g_bk,
            self.g_sk,
            self.g_h,
            self.g_l,
        )
        return (
            _voltage(self.v)
            and all(_probability(gate) for gate in gates)
            and all(math.isfinite(g) and g >= 0.0 for g in conductances)
            and _all_finite(
                self.ca,
                self.e_na,
                self.e_k,
                self.e_ca,
                self.e_h,
                self.e_l,
                self.c_m,
                self.tau_ca,
                self.kd_bk,
                self.kd_sk,
                self.dt,
                self.gain,
            )
            and self.ca >= 0.0
            and self.c_m > 0.0
            and self.tau_ca > 0.0
            and self.kd_bk > 0.0
            and self.kd_sk > 0.0
            and self.dt > 0.0
            and self.sub_steps > 0
            and self.gain >= 0.0
        )

    def step(self, current: float = 0.0) -> int:
        if not math.isfinite(current) or not self._valid_state():
            return 0

        input_current = self.gain * current
        dt_sub = self.dt / self.sub_steps
        v_prev = self.v
        v = self.v
        m = self.m
        h = self.h
        p_na = self.p_na
        n = self.n
        a = self.a
        b = self.b
        w = self.w
        m_t = self.m_t
        s = self.s
        c_n = self.c_n
        r = self.r
        ca = self.ca

        for _ in range(self.sub_steps):
            alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            beta_m = 4.0 * math.exp(-(v + 60.0) / 18.0)
            alpha_h = 0.07 * math.exp(-(v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))
            m_next = _gate_alpha_beta(m, alpha_m, beta_m, 5.0, dt_sub)
            h_next = _gate_alpha_beta(h, alpha_h, beta_h, 5.0, dt_sub)
            if m_next is None or h_next is None:
                return 0

            pna_inf = _boltz(v, -48.0, 5.0)
            tau_pna = 5.0 + 20.0 / max(0.01, 1.0 + ((v + 48.0) / 10.0) ** 2)
            p_na_next = _gate_inf(p_na, pna_inf, tau_pna, dt_sub)
            if p_na_next is None:
                return 0

            alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            beta_n = 0.125 * math.exp(-(v + 44.0) / 80.0)
            n_next = _gate_alpha_beta(n, alpha_n, beta_n, 5.0, dt_sub)
            if n_next is None:
                return 0

            a_inf = _boltz(v, -27.0, 16.0)
            b_inf = _boltz(v, -80.0, -6.0)
            a_next = _gate_inf(a, a_inf, 2.0, dt_sub)
            b_next = _gate_inf(b, b_inf, 15.0, dt_sub)
            if a_next is None or b_next is None:
                return 0

            w_inf = _boltz(v, -35.0, 10.0)
            tau_w = 100.0 / (3.3 * math.exp((v + 35.0) / 20.0) + math.exp(-(v + 35.0) / 20.0))
            w_next = _gate_inf(w, w_inf, tau_w, dt_sub)
            if w_next is None:
                return 0

            mt_inf = _boltz(v, -52.0, 5.0)
            s_inf = _boltz(v, -60.0, -6.5)
            tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0) ** 2)
            m_t_next = _gate_inf(m_t, mt_inf, 1.0, dt_sub)
            s_next = _gate_inf(s, s_inf, tau_s, dt_sub)
            if m_t_next is None or s_next is None:
                return 0

            cn_inf = _boltz(v, -20.0, 5.0)
            tau_cn = 2.0 + 10.0 / max(0.01, 1.0 + ((v + 20.0) / 10.0) ** 2)
            c_n_next = _gate_inf(c_n, cn_inf, tau_cn, dt_sub)
            if c_n_next is None:
                return 0

            r_inf = _boltz(v, -80.0, -10.0)
            tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0) ** 2)
            r_next = _gate_inf(r, r_inf, tau_r, dt_sub)
            if r_next is None:
                return 0

            g_cat = self.g_cat * m_t_next**2 * s_next
            g_can = self.g_can * c_n_next**2
            i_cat = g_cat * (v - self.e_ca)
            i_can = g_can * (v - self.e_ca)
            ca_entry = -(i_cat + i_can) * 0.001 if (i_cat + i_can) < 0.0 else 0.0
            ca_next = _calcium_exact(ca, ca_entry, self.tau_ca, dt_sub)
            if ca_next is None:
                return 0

            ca2 = ca_next**2
            kd2 = self.kd_bk**2
            bk_v = _boltz(v, 100.0 - 120.0 * ca2 / (ca2 + kd2), 15.0)
            sk_inf = ca2 / (ca2 + self.kd_sk**2)

            g_na = self.g_na_t * m_next**3 * h_next + self.g_na_p * p_na_next
            g_k = (
                self.g_kdr * n_next**4
                + self.g_ka * a_next**3 * b_next
                + self.g_km * w_next
                + self.g_bk * bk_v
                + self.g_sk * sk_inf
            )
            g_ca = g_cat + g_can
            g_h = self.g_h * r_next
            g_total = g_na + g_k + g_ca + g_h + self.g_l
            if not math.isfinite(g_total) or g_total <= 0.0:
                return 0
            steady_v = (
                input_current
                + g_na * self.e_na
                + g_k * self.e_k
                + g_ca * self.e_ca
                + g_h * self.e_h
                + self.g_l * self.e_l
            ) / g_total
            v_next = steady_v + (v - steady_v) * math.exp(-(g_total / self.c_m) * dt_sub)
            if not (_voltage(v_next) and _all_finite(ca_next) and ca_next >= 0.0):
                return 0

            v = v_next
            m = m_next
            h = h_next
            p_na = p_na_next
            n = n_next
            a = a_next
            b = b_next
            w = w_next
            m_t = m_t_next
            s = s_next
            c_n = c_n_next
            r = r_next
            ca = ca_next

        self.v = v
        self.m = m
        self.h = h
        self.p_na = p_na
        self.n = n
        self.a = a
        self.b = b
        self.w = w
        self.m_t = m_t
        self.s = s
        self.c_n = c_n
        self.r = r
        self.ca = ca

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
