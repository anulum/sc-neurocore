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
    assert all(0.0 <= gate <= 1.0 for gate in (cell.m, cell.h, cell.n, cell.a, cell.b, cell.m_t, cell.s, cell.r))
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


@pytest.mark.parametrize(
    "kwargs",
    [
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
