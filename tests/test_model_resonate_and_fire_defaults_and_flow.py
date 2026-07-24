# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (defaults_and_flow) from former test_model_resonate_and_fire.py

from __future__ import annotations

from tests.model_resonate_and_fire_support import *  # noqa: F403


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
