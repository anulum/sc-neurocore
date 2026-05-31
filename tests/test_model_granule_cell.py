# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GranuleCell behavioural tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.granule_cell import GranuleCell


def _exact_relax(value: float, target: float, tau: float, dt: float) -> float:
    return target + (value - target) * math.exp(-dt / tau)


def _snapshot(cell: GranuleCell) -> tuple[float, ...]:
    return (
        cell.v,
        cell.m,
        cell.h,
        cell.n,
        cell.a,
        cell.b,
        cell.m_t,
        cell.s,
        cell.ca,
        cell.r,
    )


def test_default_step_preserves_physical_bounds() -> None:
    cell = GranuleCell()

    for _ in range(200):
        spike = cell.step(0.0)

    assert spike in (0, 1)
    assert -100.0 <= cell.v <= 60.0
    assert all(
        0.0 <= gate <= 1.0
        for gate in (cell.m, cell.h, cell.n, cell.a, cell.b, cell.m_t, cell.s, cell.r)
    )
    assert cell.ca >= 0.0


def test_dangelo_current_surface_is_present() -> None:
    cell = GranuleCell()

    conductances = {
        "g_na": cell.g_na,
        "g_kdr": cell.g_kdr,
        "g_ka": cell.g_ka,
        "g_t": cell.g_t,
        "g_kca": cell.g_kca,
        "g_h": cell.g_h,
        "g_l": cell.g_l,
        "g_tonic": cell.g_tonic,
    }

    assert all(value > 0.0 for value in conductances.values())
    assert cell.e_na > cell.v
    assert cell.e_ca > cell.v
    assert cell.e_k < cell.v
    assert cell.e_gaba < cell.v


def test_tonic_gaba_suppresses_sufficient_drive() -> None:
    with_gaba = GranuleCell()
    without_gaba = GranuleCell(g_tonic=0.0)
    spikes_with = 0
    spikes_without = 0

    for _ in range(10_000):
        spikes_with += with_gaba.step(8.0)
        spikes_without += without_gaba.step(8.0)

    assert spikes_without > spikes_with


def test_t_type_gate_remains_deinactivated_at_rest() -> None:
    cell = GranuleCell()

    for _ in range(1000):
        cell.step(0.0)

    assert cell.s > 0.5


def test_gate_and_calcium_kinetics_use_closed_form_relaxation() -> None:
    cell = GranuleCell(
        g_na=0.0,
        g_kdr=0.0,
        g_ka=0.0,
        g_t=0.0,
        g_kca=0.0,
        g_h=0.0,
        g_l=0.0,
        g_tonic=0.0,
        gain=0.0,
        sub_steps=1,
    )
    before = _snapshot(cell)
    v0 = cell.v

    m_inf = cell._boltz(v0, -30.0, 7.0)
    tau_m = 0.1 + 0.3 / max(0.01, 1.0 + ((v0 + 30.0) / 10.0) ** 2)
    h_inf = cell._boltz(v0, -52.0, -6.0)
    tau_h = 0.5 + 5.0 / max(0.01, 1.0 + ((v0 + 50.0) / 15.0) ** 2)
    n_inf = cell._boltz(v0, -35.0, 8.0)
    tau_n = 1.0 + 5.0 / max(0.01, 1.0 + ((v0 + 35.0) / 15.0) ** 2)
    a_inf = cell._boltz(v0, -50.0, 20.0)
    b_inf = cell._boltz(v0, -70.0, -6.0)
    mt_inf = cell._boltz(v0, -52.0, 5.0)
    s_inf = cell._boltz(v0, -60.0, -6.5)
    tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v0 + 65.0) / 10.0) ** 2)
    r_inf = cell._boltz(v0, -80.0, -10.0)
    tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v0 + 80.0) / 20.0) ** 2)

    expected = (
        v0,
        _exact_relax(before[1], m_inf, tau_m, cell.dt),
        _exact_relax(before[2], h_inf, tau_h, cell.dt),
        _exact_relax(before[3], n_inf, tau_n, cell.dt),
        _exact_relax(before[4], a_inf, 2.0, cell.dt),
        _exact_relax(before[5], b_inf, 50.0, cell.dt),
        _exact_relax(before[6], mt_inf, 1.0, cell.dt),
        _exact_relax(before[7], s_inf, tau_s, cell.dt),
        _exact_relax(before[8], 0.0, cell.tau_ca, cell.dt),
        _exact_relax(before[9], r_inf, tau_r, cell.dt),
    )

    cell.step(0.0)

    for observed, expected_value in zip(_snapshot(cell), expected, strict=True):
        assert observed == pytest.approx(expected_value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"v": -100.1},
        {"v": 60.1},
        {"m": -0.1},
        {"h": 1.1},
        {"ca": -1e-6},
        {"g_na": -1.0},
        {"g_kdr": -1.0},
        {"g_ka": -1.0},
        {"g_t": -1.0},
        {"g_kca": -1.0},
        {"g_h": -1.0},
        {"g_l": -1.0},
        {"g_tonic": -1.0},
        {"c_m": 0.0},
        {"tau_ca": 0.0},
        {"kd_kca": 0.0},
        {"dt": 0.0},
        {"sub_steps": 0},
        {"gain": -1.0},
        {"v": math.inf},
    ],
)
def test_invalid_configuration_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GranuleCell(**kwargs)


def test_nonfinite_current_does_not_mutate_state() -> None:
    cell = GranuleCell()
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        cell.step(math.nan)

    assert _snapshot(cell) == before


def test_corrupted_runtime_state_does_not_mutate_state() -> None:
    cell = GranuleCell()
    cell.m = -0.1
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        cell.step(8.0)

    assert _snapshot(cell) == before
