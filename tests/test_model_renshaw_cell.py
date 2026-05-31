# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore Renshaw cell behavioral tests

from __future__ import annotations

import math
from dataclasses import asdict

from sc_neurocore.neurons.models.renshaw_cell import RenshawCell


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


def _exact_gate(previous: float, alpha: float, beta: float, phi: float, dt: float) -> float:
    total = phi * (alpha + beta)
    steady = alpha / (alpha + beta)
    return min(1.0, max(0.0, steady + (previous - steady) * math.exp(-total * dt)))


def _exact_adapt(previous: float, steady: float, tau: float, dt: float) -> float:
    return min(1.0, max(0.0, steady + (previous - steady) * math.exp(-dt / tau)))


def _reference_step(cell: RenshawCell, current: float) -> RenshawCell:
    n_sub = max(1, int(0.5 / max(cell.dt, 0.001)))
    v = cell.v
    h = cell.h
    n = cell.n
    adapt = cell.adapt

    for _ in range(n_sub):
        am = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        bm = 4.0 * math.exp(-(v + 60.0) / 18.0)
        ah = 0.07 * math.exp(-(v + 58.0) / 20.0)
        bh = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))
        an = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        bn = 0.125 * math.exp(-(v + 44.0) / 80.0)

        m_inf = am / (am + bm)
        h = _exact_gate(h, ah, bh, cell.phi, cell.dt)
        n = _exact_gate(n, an, bn, cell.phi, cell.dt)
        adapt_inf = 1.0 / (1.0 + math.exp(-(v + 30.0) / 5.0))
        adapt = _exact_adapt(adapt, adapt_inf, cell.tau_adapt, cell.dt)

        g_na = cell.g_na * m_inf**3 * h
        g_k = cell.g_k * n**4
        g_adapt = cell.g_adapt * adapt
        g_total = g_na + g_k + g_adapt + cell.g_l
        steady_v = (
            current + g_na * cell.e_na + g_k * cell.e_k + g_adapt * cell.e_k + cell.g_l * cell.e_l
        ) / g_total
        v = steady_v + (v - steady_v) * math.exp(-(g_total / cell.c_m) * cell.dt)

    return RenshawCell(v=v, h=h, n=n, adapt=adapt)


def test_renshaw_exact_gate_and_conductance_membrane_step() -> None:
    cell = RenshawCell()
    expected = _reference_step(RenshawCell(), 4.0)

    spike = cell.step(4.0)

    assert spike == 0
    assert math.isclose(cell.v, expected.v, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cell.h, expected.h, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cell.n, expected.n, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cell.adapt, expected.adapt, rel_tol=0.0, abs_tol=1e-12)


def test_renshaw_rejects_invalid_current_without_state_mutation() -> None:
    cell = RenshawCell()
    for _ in range(20):
        cell.step(4.0)
    before = asdict(cell)

    assert cell.step(math.nan) == 0
    assert asdict(cell) == before
    assert cell.step(math.inf) == 0
    assert asdict(cell) == before


def test_renshaw_rejects_excess_current_without_state_corruption() -> None:
    cell = RenshawCell()
    before = asdict(cell)

    assert cell.step(1.0e8) == 0

    assert asdict(cell) == before


def test_renshaw_adaptation_remains_bounded_and_increases() -> None:
    cell = RenshawCell()
    baseline = cell.adapt

    spikes = sum(cell.step(4.0) for _ in range(3000))

    assert spikes > 0
    assert cell.adapt > baseline + 0.01
    assert 0.0 <= cell.h <= 1.0
    assert 0.0 <= cell.n <= 1.0
    assert 0.0 <= cell.adapt <= 1.0
    assert -150.0 <= cell.v <= 100.0
