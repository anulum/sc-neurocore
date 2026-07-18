# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire source dynamics

"""Exercise source equations without claiming a paper-figure reproduction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron


def test_constant_input_equilibrium_matches_complex_linear_system() -> None:
    b = -0.8
    omega = 4.5
    current = 2.25
    denominator = b * b + omega * omega
    expected = (-b * current / denominator, omega * current / denominator)
    neuron = ResonateAndFireNeuron(
        x=expected[0],
        y=expected[1],
        b=b,
        omega=omega,
        threshold=100.0,
        dt=0.3,
    )
    for _ in range(20):
        assert neuron.step(current) == 0
    assert neuron.x == pytest.approx(expected[0], abs=1.0e-15)
    assert neuron.y == pytest.approx(expected[1], abs=1.0e-15)


def test_zero_input_homogeneous_flow_has_exact_radial_envelope() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.3,
        y=-0.4,
        b=-0.7,
        omega=3.0,
        threshold=100.0,
        dt=0.2,
    )
    initial_radius = math.hypot(neuron.x, neuron.y)
    radii = []
    for step in range(1, 11):
        neuron.step(0.0)
        radii.append(math.hypot(neuron.x, neuron.y))
        assert radii[-1] == pytest.approx(
            initial_radius * math.exp(neuron.b * neuron.dt * step),
            abs=1.0e-13,
        )
    assert all(left > right for left, right in zip(radii, radii[1:]))


def test_angular_frequency_controls_subthreshold_phase() -> None:
    slow = ResonateAndFireNeuron(
        x=1.0,
        b=0.0,
        omega=1.0,
        threshold=100.0,
        dt=0.01,
    )
    fast = ResonateAndFireNeuron(
        x=1.0,
        b=0.0,
        omega=3.0,
        threshold=100.0,
        dt=0.01,
    )
    slow_crossings = 0
    fast_crossings = 0
    slow_previous = slow.y
    fast_previous = fast.y
    for _ in range(2_000):
        slow.step(0.0)
        fast.step(0.0)
        slow_crossings += int(slow_previous <= 0.0 < slow.y)
        fast_crossings += int(fast_previous <= 0.0 < fast.y)
        slow_previous = slow.y
        fast_previous = fast.y
    assert fast_crossings > 2 * slow_crossings


def test_varied_real_drive_changes_both_coordinates() -> None:
    steps = 2_000
    constant = ResonateAndFireNeuron(threshold=100.0)
    varied = ResonateAndFireNeuron(threshold=100.0)
    constant_trace = []
    varied_trace = []
    for index in range(steps):
        constant.step(3.0)
        varied.step(3.0 + 0.8 * math.sin(index * 0.017))
        constant_trace.append((constant.x, constant.y))
        varied_trace.append((varied.x, varied.y))
    constant_array = np.asarray(constant_trace)
    varied_array = np.asarray(varied_trace)
    assert np.isfinite(constant_array).all()
    assert np.isfinite(varied_array).all()
    assert not np.array_equal(constant_array[:, 0], varied_array[:, 0])
    assert not np.array_equal(constant_array[:, 1], varied_array[:, 1])


def test_event_vector_is_sampled_upward_y_crossing_not_radius_level() -> None:
    neuron = ResonateAndFireNeuron(
        x=2.0,
        y=0.0,
        b=-0.2,
        omega=2.0,
        threshold=1.0,
        dt=0.05,
    )
    events = []
    states = []
    for _ in range(200):
        events.append(neuron.step(0.0))
        states.append((neuron.x, neuron.y))
    assert events[0] == 0
    assert sum(events) > 0
    for event, state in zip(events, states, strict=True):
        if event:
            assert state == (0.0, 1.0)


def test_long_source_default_regime_is_finite_and_deterministic() -> None:
    traces = []
    for _ in range(2):
        neuron = ResonateAndFireNeuron()
        rows = []
        for index in range(10_000):
            event = neuron.step(4.0 + 1.2 * math.sin(index * 0.0037))
            rows.append((neuron.x, neuron.y, event))
        assert math.isfinite(neuron.x) and math.isfinite(neuron.y)
        traces.append(rows)
    np.testing.assert_array_equal(traces[0], traces[1])
