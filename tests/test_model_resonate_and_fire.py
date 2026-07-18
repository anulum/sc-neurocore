# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-contract tests for ResonateAndFireNeuron

"""Verify the Izhikevich (2001) complex resonator and maintained event rule."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron


def _exact_flow(
    x: float,
    y: float,
    current: float,
    b: float,
    omega: float,
    dt: float,
) -> tuple[float, float]:
    """Independent closed-form constant-real-input flow."""
    denominator = b * b + omega * omega
    x_ss = -b * current / denominator
    y_ss = omega * current / denominator
    decay = math.exp(b * dt)
    angle = omega * dt
    dx = x - x_ss
    dy = y - y_ss
    return (
        x_ss + decay * (dx * math.cos(angle) - dy * math.sin(angle)),
        y_ss + decay * (dx * math.sin(angle) + dy * math.cos(angle)),
    )


def test_defaults_match_the_source_illustration_and_engineering_step() -> None:
    neuron = ResonateAndFireNeuron()
    assert (neuron.x, neuron.y) == (0.0, 0.0)
    assert (neuron.b, neuron.omega, neuron.threshold, neuron.dt) == (
        -1.0,
        10.0,
        1.0,
        0.01,
    )


def test_constructor_normalises_numeric_scalars() -> None:
    neuron = ResonateAndFireNeuron(
        x=1,
        y=cast(float, "-0.25"),
        b=-2,
        omega=5,
        threshold=3,
        dt=cast(float, "0.02"),
    )
    assert (neuron.x, neuron.y, neuron.b, neuron.omega, neuron.threshold, neuron.dt) == (
        1.0,
        -0.25,
        -2.0,
        5.0,
        3.0,
        0.02,
    )


def test_one_step_matches_independent_exact_linear_flow() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.3,
        y=-0.2,
        b=-0.2,
        omega=1.7,
        threshold=100.0,
        dt=1.25,
    )
    expected = _exact_flow(
        neuron.x,
        neuron.y,
        0.8,
        neuron.b,
        neuron.omega,
        neuron.dt,
    )
    assert neuron.step(0.8) == 0
    assert neuron.x == pytest.approx(expected[0], rel=0.0, abs=1.0e-12)
    assert neuron.y == pytest.approx(expected[1], rel=0.0, abs=1.0e-12)


def test_equilibrium_is_a_fixed_point_for_constant_input() -> None:
    b = -0.7
    omega = 3.2
    current = 1.4
    denominator = b * b + omega * omega
    x_ss = -b * current / denominator
    y_ss = omega * current / denominator
    neuron = ResonateAndFireNeuron(
        x=x_ss,
        y=y_ss,
        b=b,
        omega=omega,
        threshold=100.0,
        dt=0.37,
    )
    assert neuron.step(current) == 0
    assert neuron.x == pytest.approx(x_ss, rel=0.0, abs=1.0e-15)
    assert neuron.y == pytest.approx(y_ss, rel=0.0, abs=1.0e-15)


def test_homogeneous_radius_decays_by_exp_b_dt() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.6,
        y=-0.8,
        b=-0.5,
        omega=4.0,
        threshold=100.0,
        dt=2.0,
    )
    initial_radius = math.hypot(neuron.x, neuron.y)
    assert neuron.step(0.0) == 0
    assert math.hypot(neuron.x, neuron.y) == pytest.approx(
        initial_radius * math.exp(-1.0),
        rel=0.0,
        abs=1.0e-12,
    )


def test_omega_sets_the_exact_rotation_angle() -> None:
    omega = 2.0
    neuron = ResonateAndFireNeuron(
        x=1.0,
        y=0.0,
        b=0.0,
        omega=omega,
        threshold=10.0,
        dt=math.pi / (2.0 * omega),
    )
    assert neuron.step(0.0) == 0
    assert neuron.x == pytest.approx(0.0, abs=1.0e-15)
    assert neuron.y == pytest.approx(1.0, abs=1.0e-15)


def test_spike_is_upward_y_crossing_and_installs_source_reset() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.0,
        y=0.99,
        b=0.0,
        omega=1.0,
        threshold=1.0,
        dt=0.1,
    )
    assert neuron.step(10.0) == 1
    assert (neuron.x, neuron.y) == (0.0, 1.0)


def test_source_reset_at_threshold_does_not_immediately_retrigger() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.0,
        y=0.99,
        b=0.0,
        omega=1.0,
        threshold=1.0,
        dt=0.1,
    )
    assert neuron.step(10.0) == 1
    assert neuron.step(0.0) == 0
    assert neuron.y < neuron.threshold


def test_radius_above_threshold_is_not_itself_an_event() -> None:
    neuron = ResonateAndFireNeuron(
        x=2.0,
        y=0.0,
        b=-1.0,
        omega=1.0e-9,
        threshold=1.0,
        dt=0.01,
    )
    assert math.hypot(neuron.x, neuron.y) > neuron.threshold
    assert neuron.step(0.0) == 0


def test_downward_y_crossing_is_not_an_event() -> None:
    neuron = ResonateAndFireNeuron(
        x=-1.0,
        y=1.01,
        b=0.0,
        omega=1.0,
        threshold=1.0,
        dt=0.02,
    )
    assert neuron.step(0.0) == 0
    assert neuron.y < 1.01


def test_zero_current_quiescent_state_remains_quiescent() -> None:
    neuron = ResonateAndFireNeuron()
    assert [neuron.step(0.0) for _ in range(100)] == [0] * 100
    assert (neuron.x, neuron.y) == (0.0, 0.0)


@pytest.mark.parametrize(("current", "expected_spikes"), ((5.0, 0), (10.0, 15)))
def test_source_default_constant_drive_regimes(current: float, expected_spikes: int) -> None:
    """Separate the source-default subthreshold and spiking regimes."""
    neuron = ResonateAndFireNeuron()
    assert sum(neuron.step(current) for _ in range(500)) == expected_spikes


def test_long_varied_run_is_finite_and_deterministic() -> None:
    drive = 4.0 + 1.2 * np.sin(np.arange(20_000, dtype=np.float64) * 0.013)
    first = ResonateAndFireNeuron()
    second = ResonateAndFireNeuron()
    first_trace = [(first.step(value), first.x, first.y) for value in drive]
    second_trace = [(second.step(value), second.x, second.y) for value in drive]
    assert first_trace == second_trace
    assert np.isfinite((first.x, first.y)).all()


def test_python_batch_matches_scalar_complete_trace() -> None:
    drive = 3.0 + np.sin(np.arange(256, dtype=np.float64) * 0.071)
    batch_neuron = ResonateAndFireNeuron(x=0.15, y=-0.2)
    scalar_neuron = ResonateAndFireNeuron(x=0.15, y=-0.2)
    result = batch_neuron.simulate(drive, backend="python")

    x_expected: list[float] = []
    y_expected: list[float] = []
    spikes_expected: list[float] = []
    for value in drive:
        spikes_expected.append(float(scalar_neuron.step(float(value))))
        x_expected.append(scalar_neuron.x)
        y_expected.append(scalar_neuron.y)

    np.testing.assert_array_equal(result["x"], np.asarray(x_expected))
    np.testing.assert_array_equal(result["y"], np.asarray(y_expected))
    np.testing.assert_array_equal(result["spikes"], np.asarray(spikes_expected))
    assert result["spike_count"] == int(sum(spikes_expected))
    assert (batch_neuron.x, batch_neuron.y) == (scalar_neuron.x, scalar_neuron.y)


def test_empty_batch_preserves_initial_state() -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    result = neuron.simulate([], backend="python")
    assert np.asarray(result["x"]).shape == (0,)
    assert np.asarray(result["y"]).shape == (0,)
    assert np.asarray(result["spikes"]).shape == (0,)
    assert result["x_final"] == 0.25
    assert result["y_final"] == -0.5
    assert result["spike_count"] == 0
    assert (neuron.x, neuron.y) == (0.25, -0.5)


def test_reset_restores_quiescent_state_and_preserves_parameters() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.5,
        y=-0.25,
        b=-0.5,
        omega=2.0,
        threshold=3.0,
        dt=0.02,
    )
    neuron.reset()
    assert (neuron.x, neuron.y) == (0.0, 0.0)
    assert (neuron.b, neuron.omega, neuron.threshold, neuron.dt) == (
        -0.5,
        2.0,
        3.0,
        0.02,
    )


@pytest.mark.parametrize("field", ("x", "y", "b", "omega", "threshold", "dt"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_constructor_rejects_nonfinite_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ResonateAndFireNeuron(**{field: value})


@pytest.mark.parametrize("field", ("omega", "threshold", "dt"))
@pytest.mark.parametrize("value", (0.0, -1.0))
def test_constructor_rejects_nonpositive_scales(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ResonateAndFireNeuron(**{field: value})


@pytest.mark.parametrize("field", ("x", "y", "b", "omega", "threshold", "dt"))
def test_constructor_rejects_nonnumeric_values(field: str) -> None:
    with pytest.raises(ValueError, match="numeric"):
        ResonateAndFireNeuron(**cast(dict[str, float], {field: object()}))


@pytest.mark.parametrize("current", (np.nan, np.inf, -np.inf, object()))
def test_invalid_current_is_atomic(current: object) -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="current"):
        neuron.step(cast(float, current))
    assert (neuron.x, neuron.y) == before


def test_corrupted_runtime_configuration_is_atomic() -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    neuron.dt = 0.0
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="dt"):
        neuron.step(0.5)
    assert (neuron.x, neuron.y) == before


@pytest.mark.parametrize("field", ("x", "y"))
def test_corrupted_runtime_state_type_is_atomic(field: str) -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    setattr(neuron, field, object())
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="state must be numeric"):
        neuron.step(0.5)
    assert (neuron.x, neuron.y) == before


@pytest.mark.parametrize("field", ("b", "omega", "threshold", "dt"))
def test_corrupted_runtime_parameter_type_is_atomic(field: str) -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    setattr(neuron, field, object())
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="parameters must be numeric"):
        neuron.step(0.5)
    assert (neuron.x, neuron.y) == before


def test_nonfinite_exact_candidate_is_atomic() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.25,
        y=-0.5,
        b=1.0e308,
        omega=1.0,
        threshold=1.0e308,
        dt=1.0e308,
    )
    before = (neuron.x, neuron.y)
    with pytest.raises(FloatingPointError, match="coefficients"):
        neuron.step(1.0e308)
    assert (neuron.x, neuron.y) == before


def test_batch_rejects_nonfinite_drive_before_mutation() -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="finite"):
        neuron.simulate([0.0, np.nan, 1.0], backend="python")
    assert (neuron.x, neuron.y) == before


def test_batch_rejects_unknown_backend_before_mutation() -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="unknown resonate-and-fire backend"):
        neuron.simulate([0.0], backend="fortran")
    assert (neuron.x, neuron.y) == before


def test_exact_flow_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="denominator"):
        ResonateAndFireNeuron._exact_linear_flow(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )


def test_exact_flow_rejects_exponential_overflow() -> None:
    with pytest.raises(FloatingPointError, match="decay"):
        ResonateAndFireNeuron._exact_linear_flow(
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1000.0,
        )


def test_exact_flow_rejects_nonfinite_equilibrium() -> None:
    with pytest.raises(FloatingPointError, match="equilibrium"):
        ResonateAndFireNeuron._exact_linear_flow(
            0.0,
            0.0,
            1.0e308,
            1.0e-154,
            1.0e-154,
            0.01,
        )


def test_exact_flow_rejects_nonfinite_post_rotation_candidate() -> None:
    with pytest.raises(FloatingPointError, match="candidate"):
        ResonateAndFireNeuron._exact_linear_flow(
            1.0e308,
            1.0e308,
            0.0,
            1.0,
            1.0,
            1.0,
        )
