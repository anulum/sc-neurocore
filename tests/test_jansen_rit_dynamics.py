# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Jansen–Rit equation and dynamics tests

"""Exercise source parameterisation without claiming a figure reproduction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit


def test_default_inverse_rates_equal_source_time_constants() -> None:
    unit = JansenRitUnit()
    assert 1.0 / unit.a_rate == pytest.approx(0.010)
    assert 1.0 / unit.b_rate == pytest.approx(0.020)


def test_half_maximum_notation_equals_source_e0_notation() -> None:
    unit = JansenRitUnit()
    voltage = 4.2
    maintained = unit._sigmoid(voltage)
    source = 5.0 / (1.0 + math.exp(0.56 * (6.0 - voltage)))
    assert maintained == pytest.approx(source)


def test_zero_state_first_step_matches_closed_form_values() -> None:
    unit = JansenRitUnit()
    base_rate = unit._sigmoid(0.0)
    unit.step(220.0)
    assert unit.y0 == 0.0
    assert unit.y3 == pytest.approx(unit.dt * unit.a_exc * unit.a_rate * base_rate)
    assert unit.y1 == 0.0
    assert unit.y4 == pytest.approx(
        unit.dt * unit.a_exc * unit.a_rate * (220.0 + 0.8 * unit.c * base_rate)
    )
    assert unit.y2 == 0.0
    assert unit.y5 == pytest.approx(unit.dt * unit.b_exc * unit.b_rate * 0.25 * unit.c * base_rate)


def test_varied_drive_changes_complete_trajectory() -> None:
    steps = 5_000
    constant = JansenRitUnit()
    varied = JansenRitUnit()
    constant_trace = np.asarray([constant.step(220.0) for _ in range(steps)])
    varied_trace = np.asarray(
        [varied.step(220.0 + 100.0 * math.sin(index * 0.0037)) for index in range(steps)]
    )
    assert np.isfinite(constant_trace).all()
    assert np.isfinite(varied_trace).all()
    assert not np.array_equal(constant_trace, varied_trace)


def test_output_is_continuous_eeg_proxy_not_binary_spike() -> None:
    unit = JansenRitUnit()
    trace = np.asarray([unit.step(220.0) for _ in range(2_000)])
    assert np.unique(trace).size > 100
    assert not set(np.unique(trace)).issubset({0.0, 1.0})
    assert trace[-1] == pytest.approx(unit.y1 - unit.y2)


def test_smaller_euler_step_converges_over_short_window() -> None:
    coarse = JansenRitUnit(dt=0.0001)
    fine = JansenRitUnit(dt=0.00005)
    for _ in range(1_000):
        coarse.step(220.0)
    for _ in range(2_000):
        fine.step(220.0)
    coarse_state = np.asarray([coarse.y0, coarse.y3, coarse.y1, coarse.y4, coarse.y2, coarse.y5])
    fine_state = np.asarray([fine.y0, fine.y3, fine.y1, fine.y4, fine.y2, fine.y5])
    np.testing.assert_allclose(coarse_state, fine_state, rtol=0.02, atol=0.02)
