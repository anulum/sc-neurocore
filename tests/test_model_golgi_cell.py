# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore GolgiCell behavioural tests

from __future__ import annotations

import math
from dataclasses import asdict, replace

from sc_neurocore.neurons.models.golgi_cell import GolgiCell


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


def _boltz(v: float, vh: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-(v - vh) / k))


def _gate_alpha_beta(previous: float, alpha: float, beta: float, phi: float, dt: float) -> float:
    total = phi * (alpha + beta)
    steady = alpha / (alpha + beta)
    return min(1.0, max(0.0, steady + (previous - steady) * math.exp(-total * dt)))


def _gate_inf(previous: float, steady: float, tau: float, dt: float) -> float:
    return min(1.0, max(0.0, steady + (previous - steady) * math.exp(-dt / tau)))


def _calcium(previous: float, entry: float, tau: float, dt: float) -> float:
    steady = entry * tau
    return max(0.0, steady + (previous - steady) * math.exp(-dt / tau))


def _reference_step(cell: GolgiCell, current: float) -> GolgiCell:
    v = cell.v
    m = cell.m
    h = cell.h
    p_na = cell.p_na
    n = cell.n
    a = cell.a
    b = cell.b
    w = cell.w
    m_t = cell.m_t
    s = cell.s
    c_n = cell.c_n
    r = cell.r
    ca = cell.ca
    dt_sub = cell.dt / cell.sub_steps
    input_current = cell.gain * current

    for _ in range(cell.sub_steps):
        alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        beta_m = 4.0 * math.exp(-(v + 60.0) / 18.0)
        alpha_h = 0.07 * math.exp(-(v + 58.0) / 20.0)
        beta_h = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))
        m = _gate_alpha_beta(m, alpha_m, beta_m, 5.0, dt_sub)
        h = _gate_alpha_beta(h, alpha_h, beta_h, 5.0, dt_sub)

        pna_inf = _boltz(v, -48.0, 5.0)
        tau_pna = 5.0 + 20.0 / max(0.01, 1.0 + ((v + 48.0) / 10.0) ** 2)
        p_na = _gate_inf(p_na, pna_inf, tau_pna, dt_sub)

        alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        beta_n = 0.125 * math.exp(-(v + 44.0) / 80.0)
        n = _gate_alpha_beta(n, alpha_n, beta_n, 5.0, dt_sub)

        a = _gate_inf(a, _boltz(v, -27.0, 16.0), 2.0, dt_sub)
        b = _gate_inf(b, _boltz(v, -80.0, -6.0), 15.0, dt_sub)
        tau_w = 100.0 / (3.3 * math.exp((v + 35.0) / 20.0) + math.exp(-(v + 35.0) / 20.0))
        w = _gate_inf(w, _boltz(v, -35.0, 10.0), tau_w, dt_sub)
        m_t = _gate_inf(m_t, _boltz(v, -52.0, 5.0), 1.0, dt_sub)
        tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0) ** 2)
        s = _gate_inf(s, _boltz(v, -60.0, -6.5), tau_s, dt_sub)
        tau_cn = 2.0 + 10.0 / max(0.01, 1.0 + ((v + 20.0) / 10.0) ** 2)
        c_n = _gate_inf(c_n, _boltz(v, -20.0, 5.0), tau_cn, dt_sub)
        tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0) ** 2)
        r = _gate_inf(r, _boltz(v, -80.0, -10.0), tau_r, dt_sub)

        g_cat = cell.g_cat * m_t**2 * s
        g_can = cell.g_can * c_n**2
        i_cat = g_cat * (v - cell.e_ca)
        i_can = g_can * (v - cell.e_ca)
        ca_entry = -(i_cat + i_can) * 0.001 if (i_cat + i_can) < 0.0 else 0.0
        ca = _calcium(ca, ca_entry, cell.tau_ca, dt_sub)
        ca2 = ca * ca
        bk_v = _boltz(v, 100.0 - 120.0 * ca2 / (ca2 + cell.kd_bk**2), 15.0)
        sk_inf = ca2 / (ca2 + cell.kd_sk**2)

        g_na = cell.g_na_t * m**3 * h + cell.g_na_p * p_na
        g_k = (
            cell.g_kdr * n**4
            + cell.g_ka * a**3 * b
            + cell.g_km * w
            + cell.g_bk * bk_v
            + cell.g_sk * sk_inf
        )
        g_ca = g_cat + g_can
        g_h = cell.g_h * r
        g_total = g_na + g_k + g_ca + g_h + cell.g_l
        steady_v = (
            input_current
            + g_na * cell.e_na
            + g_k * cell.e_k
            + g_ca * cell.e_ca
            + g_h * cell.e_h
            + cell.g_l * cell.e_l
        ) / g_total
        v = steady_v + (v - steady_v) * math.exp(-(g_total / cell.c_m) * dt_sub)

    return replace(
        cell, v=v, m=m, h=h, p_na=p_na, n=n, a=a, b=b, w=w, m_t=m_t, s=s, c_n=c_n, r=r, ca=ca
    )


def test_golgi_exact_gate_calcium_and_conductance_step() -> None:
    cell = GolgiCell()
    expected = _reference_step(GolgiCell(), 5.0)

    spike = cell.step(5.0)

    assert spike == 0
    assert math.isclose(cell.v, expected.v, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cell.m, expected.m, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cell.h, expected.h, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cell.p_na, expected.p_na, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cell.n, expected.n, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cell.ca, expected.ca, rel_tol=0.0, abs_tol=1e-12)


def test_golgi_invalid_current_preserves_state() -> None:
    cell = GolgiCell()
    for _ in range(10):
        cell.step(5.0)
    before = asdict(cell)

    assert cell.step(math.nan) == 0
    assert asdict(cell) == before
    assert cell.step(math.inf) == 0
    assert asdict(cell) == before


def test_golgi_excess_current_preserves_state() -> None:
    cell = GolgiCell()
    before = asdict(cell)

    assert cell.step(1.0e8) == 0

    assert asdict(cell) == before


def test_golgi_all_currents_bounded_and_calcium_active() -> None:
    cell = GolgiCell()
    baseline_ca = cell.ca
    spikes = sum(cell.step(10.0) for _ in range(2000))

    assert spikes > 0
    assert cell.ca > baseline_ca
    for gate in (
        cell.m,
        cell.h,
        cell.p_na,
        cell.n,
        cell.a,
        cell.b,
        cell.w,
        cell.m_t,
        cell.s,
        cell.c_n,
        cell.r,
    ):
        assert 0.0 <= gate <= 1.0
    assert -100.0 <= cell.v <= 60.0
    assert cell.ca >= 0.0
