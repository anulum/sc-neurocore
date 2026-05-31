# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — StellateCell behavioural tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.stellate_cell import StellateCell


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    z = -d / k
    if z > 60.0:
        return 0.0
    if z < -60.0:
        return a * d
    return a * d / (1.0 - math.exp(z))


def _boltz(v: float, vh: float, k: float) -> float:
    z = -(v - vh) / k
    if z > 60.0:
        return 0.0
    if z < -60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def _safe_exp(value: float) -> float:
    return math.exp(max(-60.0, min(60.0, value)))


def _exact_hh_gate(value: float, alpha: float, beta: float, phi: float, dt: float) -> float:
    rate = phi * (alpha + beta)
    target = alpha / (alpha + beta)
    return target + (value - target) * math.exp(-rate * dt)


def _exact_relax(value: float, target: float, tau: float, dt: float) -> float:
    return target + (value - target) * math.exp(-dt / tau)


def _snapshot(cell: StellateCell) -> tuple[float, float, float, float]:
    return cell.v, cell.h, cell.n, cell.p


def test_default_step_preserves_wb_kv3_bounds() -> None:
    cell = StellateCell()

    for _ in range(200):
        spike = cell.step(0.0)

    assert spike in (0, 1)
    assert -100.0 <= cell.v <= 60.0
    assert all(0.0 <= gate <= 1.0 for gate in (cell.h, cell.n, cell.p))


def test_kv3_gate_activates_more_during_depolarisation_than_rest() -> None:
    resting = StellateCell()
    depolarised = StellateCell()

    for _ in range(100):
        resting.step(0.0)
        depolarised.step(8.0)

    assert depolarised.p > resting.p


def test_gate_kinetics_use_closed_form_relaxation() -> None:
    cell = StellateCell(g_na=0.0, g_k=0.0, g_kv3=0.0, g_l=0.0, gain=0.0, _sub_steps=1)
    before = _snapshot(cell)
    v0 = cell.v

    alpha_h = 0.07 * _safe_exp(-(v0 + 58.0) / 20.0)
    beta_h = _boltz(v0, -28.0, 10.0)
    alpha_n = _safe_rate(0.01, 34.0, v0, 10.0, 0.1)
    beta_n = 0.125 * _safe_exp(-(v0 + 44.0) / 80.0)
    p_inf = _boltz(v0, -10.0, 10.0)
    tau_p = 1.0 + 4.0 / (1.0 + _safe_exp((v0 + 20.0) / 15.0))

    expected = (
        v0,
        _exact_hh_gate(before[1], alpha_h, beta_h, cell.phi, cell.dt),
        _exact_hh_gate(before[2], alpha_n, beta_n, cell.phi, cell.dt),
        _exact_relax(before[3], p_inf, tau_p, cell.dt),
    )

    cell.step(0.0)

    for observed, expected_value in zip(_snapshot(cell), expected, strict=True):
        assert observed == pytest.approx(expected_value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"v": -100.1},
        {"v": 60.1},
        {"h": -0.1},
        {"n": 1.1},
        {"p": -0.1},
        {"g_na": -1.0},
        {"g_k": -1.0},
        {"g_kv3": -1.0},
        {"g_l": -1.0},
        {"c_m": 0.0},
        {"phi": 0.0},
        {"dt": 0.0},
        {"_sub_steps": 0},
        {"gain": -1.0},
        {"v": math.inf},
    ],
)
def test_invalid_configuration_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StellateCell(**kwargs)


def test_nonfinite_current_does_not_mutate_state() -> None:
    cell = StellateCell()
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        cell.step(math.nan)

    assert _snapshot(cell) == before


def test_corrupted_runtime_state_does_not_mutate_state() -> None:
    cell = StellateCell()
    cell.h = -0.1
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        cell.step(8.0)

    assert _snapshot(cell) == before
