# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Jansen–Rit scalar and atomic-batch contracts

"""Verify equation-(6) state, safety, and public simulation semantics."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit

_STATE_NAMES = ("y0", "y3", "y1", "y4", "y2", "y5")


def test_published_defaults_and_source_bound_euler_step() -> None:
    unit = JansenRitUnit()
    assert tuple(getattr(unit, name) for name in _STATE_NAMES) == (0.0,) * 6
    assert (unit.a_exc, unit.b_exc, unit.a_rate, unit.b_rate, unit.c) == (
        3.25,
        22.0,
        100.0,
        50.0,
        135.0,
    )
    assert (unit.e0, unit.v0, unit.r, unit.dt) == (2.5, 6.0, 0.56, 0.0001)


def test_sigmoid_is_stable_and_uses_half_maximum_parameterisation() -> None:
    unit = JansenRitUnit()
    assert unit._sigmoid(unit.v0) == pytest.approx(unit.e0)
    assert unit._sigmoid(-1.0e6) == 0.0
    assert unit._sigmoid(1.0e6) == pytest.approx(2.0 * unit.e0)


def test_one_step_matches_equation_six_simultaneous_update() -> None:
    unit = JansenRitUnit(y0=0.1, y3=0.2, y1=0.3, y4=-0.4, y2=-0.1, y5=0.5)
    old = tuple(getattr(unit, name) for name in _STATE_NAMES)
    c1, c2, c3, c4 = unit.c, 0.8 * unit.c, 0.25 * unit.c, 0.25 * unit.c
    sp = unit._sigmoid(old[2] - old[4])
    se = unit._sigmoid(c1 * old[0])
    si = unit._sigmoid(c3 * old[0])
    derivatives = (
        old[1],
        unit.a_exc * unit.a_rate * sp - 2 * unit.a_rate * old[1] - unit.a_rate**2 * old[0],
        old[3],
        unit.a_exc * unit.a_rate * (220.0 + c2 * se)
        - 2 * unit.a_rate * old[3]
        - unit.a_rate**2 * old[2],
        old[5],
        unit.b_exc * unit.b_rate * c4 * si - 2 * unit.b_rate * old[5] - unit.b_rate**2 * old[4],
    )
    expected = tuple(value + unit.dt * derivative for value, derivative in zip(old, derivatives))
    eeg = unit.step(220.0)
    assert tuple(getattr(unit, name) for name in _STATE_NAMES) == pytest.approx(expected)
    assert eeg == pytest.approx(expected[2] - expected[4])


def test_c1_not_c2_is_inside_excitatory_sigmoid() -> None:
    unit = JansenRitUnit(y0=0.1)
    expected = unit.a_exc * unit.a_rate * (220.0 + 0.8 * unit.c * unit._sigmoid(unit.c * unit.y0))
    wrong = (
        unit.a_exc * unit.a_rate * (220.0 + 0.8 * unit.c * unit._sigmoid(0.8 * unit.c * unit.y0))
    )
    unit.step(220.0)
    observed = unit.y4 / unit.dt
    assert observed == pytest.approx(expected)
    assert abs(observed - wrong) > 1.0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"dt": 0.0},
        {"a_exc": 0.0},
        {"b_exc": 0.0},
        {"a_rate": 0.0},
        {"b_rate": 0.0},
        {"c": -1.0},
        {"e0": 0.0},
        {"r": 0.0},
        {"y0": math.nan},
        {"v0": math.inf},
    ),
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        JansenRitUnit(**kwargs)


def test_rejected_input_and_corrupted_state_do_not_mutate_other_fields() -> None:
    unit = JansenRitUnit()
    before = tuple(getattr(unit, name) for name in _STATE_NAMES)
    with pytest.raises(ValueError, match="p_ext"):
        unit.step(math.nan)
    assert tuple(getattr(unit, name) for name in _STATE_NAMES) == before

    unit.y4 = math.inf
    corrupted = tuple(getattr(unit, name) for name in _STATE_NAMES)
    with pytest.raises(ValueError, match="y4"):
        unit.step(220.0)
    assert tuple(getattr(unit, name) for name in _STATE_NAMES) == corrupted


def test_reset_preserves_parameters_and_clears_six_states() -> None:
    unit = JansenRitUnit(c=270.0, dt=0.0002)
    for _ in range(100):
        unit.step(220.0)
    unit.reset()
    assert tuple(getattr(unit, name) for name in _STATE_NAMES) == (0.0,) * 6
    assert (unit.c, unit.dt) == (270.0, 0.0002)


def test_long_default_trajectory_is_finite_and_deterministic() -> None:
    traces = []
    for _ in range(2):
        unit = JansenRitUnit()
        trace = [unit.step(220.0 + 100.0 * math.sin(index * 0.0037)) for index in range(20_000)]
        assert all(math.isfinite(getattr(unit, name)) for name in _STATE_NAMES)
        traces.append(trace)
    np.testing.assert_array_equal(traces[0], traces[1])


def test_python_batch_returns_all_states_and_consistent_finals() -> None:
    drive = np.linspace(120.0, 320.0, 64)
    unit = JansenRitUnit(y0=0.1, y3=0.2, y1=0.3, y4=-0.4, y2=-0.1, y5=0.5)
    result = unit.simulate(drive, backend="python")
    for key in (*_STATE_NAMES, "eeg"):
        trace = cast(npt.NDArray[np.float64], result[key])
        assert trace.shape == (64,)
        assert np.isfinite(trace).all()
    np.testing.assert_array_equal(result["eeg"], result["y1"] - result["y2"])
    for key in _STATE_NAMES:
        assert getattr(unit, key) == result[f"{key}_final"]


def test_empty_batch_preserves_initial_state() -> None:
    unit = JansenRitUnit(y0=0.1, y3=0.2, y1=0.3, y4=-0.4, y2=-0.1, y5=0.5)
    result = unit.simulate([], backend="python")
    assert all(
        cast(npt.NDArray[np.float64], result[key]).shape == (0,) for key in (*_STATE_NAMES, "eeg")
    )
    assert tuple(getattr(unit, name) for name in _STATE_NAMES) == (0.1, 0.2, 0.3, -0.4, -0.1, 0.5)
