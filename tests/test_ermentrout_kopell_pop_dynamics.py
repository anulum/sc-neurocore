# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MPR equation-(12) dynamics tests

"""Exercise source parameterisation without claiming a figure reproduction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.ermentrout_kopell_pop import (
    ErmentroutKopellPopulation,
)


def test_lorentzian_width_injects_rate_at_zero_population_state() -> None:
    unit = ErmentroutKopellPopulation(r=0.0, v=0.0, delta=0.7, tau=2.0, dt=0.005)
    unit.step(0.0)
    assert unit.r == pytest.approx(0.005 * 0.7 / (math.pi * 4.0))


def test_simultaneous_update_uses_pre_step_rate_in_voltage_equation() -> None:
    unit = ErmentroutKopellPopulation(r=0.2, v=-1.5, delta=0.7, j=12.0, dt=0.01)
    old_r, old_v = unit.r, unit.v
    drive = 1.25
    expected_v = old_v + unit.dt * (
        old_v**2 + unit.eta_bar + drive + unit.j * old_r - (math.pi * old_r) ** 2
    )
    unit.step(drive)
    assert unit.v == expected_v


def test_varied_drive_changes_complete_trajectory() -> None:
    steps = 2_000
    constant = ErmentroutKopellPopulation()
    varied = ErmentroutKopellPopulation()
    constant_trace = []
    varied_trace = []
    for index in range(steps):
        constant.step(1.5)
        varied.step(1.5 + 0.5 * math.sin(index * 0.017))
        constant_trace.append((constant.r, constant.v))
        varied_trace.append((varied.r, varied.v))
    constant_array = np.asarray(constant_trace)
    varied_array = np.asarray(varied_trace)
    assert np.isfinite(constant_array).all()
    assert np.isfinite(varied_array).all()
    assert not np.array_equal(constant_array, varied_array)


def test_output_is_continuous_rate_not_binary_spike() -> None:
    unit = ErmentroutKopellPopulation()
    trace = np.asarray([unit.step(1.5) for _ in range(1_000)])
    assert np.unique(trace).size > 100
    assert not set(np.unique(trace)).issubset({0.0, 1.0})
    assert trace[-1] == unit.r


def test_smaller_euler_step_converges_over_short_window() -> None:
    coarse = ErmentroutKopellPopulation(dt=0.01)
    fine = ErmentroutKopellPopulation(dt=0.005)
    for index in range(100):
        coarse.step(1.5 + 0.25 * math.sin(index * 0.02))
    for index in range(200):
        fine.step(1.5 + 0.25 * math.sin((index // 2) * 0.02))
    np.testing.assert_allclose(
        np.asarray((coarse.r, coarse.v)),
        np.asarray((fine.r, fine.v)),
        rtol=0.02,
        atol=0.02,
    )


def test_long_reference_regime_is_finite_and_deterministic() -> None:
    traces = []
    for _ in range(2):
        unit = ErmentroutKopellPopulation()
        rows = []
        for index in range(10_000):
            unit.step(1.5 + 0.5 * math.sin(index * 0.0037))
            rows.append((unit.r, unit.v))
        assert math.isfinite(unit.r) and math.isfinite(unit.v)
        traces.append(rows)
    np.testing.assert_array_equal(traces[0], traces[1])
