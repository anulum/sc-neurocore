# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CochlearHairCell module tests

"""Module-specific behavioural tests for the cochlear MET hair-cell model."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.cochlear_hair_cell import CochlearHairCell


def _stable_boltzmann(displacement: float, x0: float, delta: float) -> float:
    z = (displacement - x0) / delta
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _exact_voltage(cell: CochlearHairCell, displacement: float) -> float:
    po = _stable_boltzmann(displacement, cell.x0, cell.delta)
    g_met = cell.g_max * po
    g_total = cell.g_l + g_met
    v_inf = (cell.g_l * cell.e_l + g_met * cell.e_met) / g_total
    return v_inf + (cell.v - v_inf) * math.exp(-(g_total / cell.cap) * cell.dt)


def test_cochlear_step_matches_closed_form_membrane_relaxation() -> None:
    cell = CochlearHairCell(v=-60.0, dt=0.01)
    expected = _exact_voltage(cell, displacement=0.0)

    spike = cell.step(0.0)

    assert spike in (0, 1)
    assert cell.v == pytest.approx(expected, abs=1e-12)
    assert cell.glutamate_release == pytest.approx(max(cell.v + 60.0, 0.0) / 40.0)


def test_cochlear_large_displacement_boltzmann_is_finite() -> None:
    cell = CochlearHairCell()

    assert cell.p_open(1000.0) == pytest.approx(1.0)
    assert cell.p_open(-1000.0) == pytest.approx(0.0)


def test_cochlear_rejects_invalid_runtime_parameter_without_mutation() -> None:
    cell = CochlearHairCell(v=-55.0, glutamate_release=0.125)
    before = cell.state()
    cell.cap = -1.0

    with pytest.raises(ValueError, match="cap"):
        cell.step(0.25)

    assert cell.v == before["v"]
    assert cell.glutamate_release == before["glutamate_release"]


def test_cochlear_rejects_corrupted_runtime_state_without_mutation() -> None:
    cell = CochlearHairCell(v=-55.0, glutamate_release=0.125)
    before = cell.state()
    cell.v = math.inf

    with pytest.raises(ValueError, match="v"):
        cell.step(0.25)

    assert cell.glutamate_release == before["glutamate_release"]
    assert math.isinf(cell.v)


def test_cochlear_reset_restores_leak_equilibrium() -> None:
    cell = CochlearHairCell(v=-42.0, glutamate_release=0.45)

    cell.reset()

    assert cell.v == cell.e_l
    assert cell.glutamate_release == 0.0
