# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive-threshold source-dynamics contracts

"""Contracts of the composite reduced adaptive-threshold dynamics."""

from __future__ import annotations

import math

import numpy as np

from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron


def test_membrane_relaxation_matches_linear_system_closed_form() -> None:
    """The membrane candidate equals the exact constant-input LIF solution."""
    n = AdaptiveThresholdIFNeuron(v=-58.0, theta=-40.0, tau_m=7.5, dt=0.05)
    current = 9.25
    steady = n.v_rest + current
    expected = steady + (n.v - steady) * math.exp(-n.dt / n.tau_m)
    assert n.step(current) == 0
    assert abs(n.v - expected) < 1.0e-13


def test_threshold_decay_is_the_a0_mihalas_niebur_limit() -> None:
    """dTheta/dt = -b(Theta-Theta_inf) recovered exactly with b=1/tau_theta."""
    tau_theta = 37.5
    n = AdaptiveThresholdIFNeuron(theta=-42.0, tau_theta=tau_theta, dt=0.2)
    b = 1.0 / tau_theta
    expected = n.theta + (-b * (n.theta - n.theta_rest)) * ((1.0 - math.exp(-b * n.dt)) / b)
    assert n.step(0.0) == 0
    assert abs(n.theta - expected) < 1.0e-13


def test_threshold_increases_by_a_fixed_amount_after_each_spike() -> None:
    """The Platkiewicz-Brette post-spike shift is exactly delta_theta."""
    n = AdaptiveThresholdIFNeuron(v=-50.5, theta=-51.0, delta_theta=4.25)
    relaxed = n.theta_rest + (n.theta - n.theta_rest) * math.exp(-n.dt / n.tau_theta)
    assert n.step(0.0) == 1
    assert abs(n.theta - (relaxed + 4.25)) < 1.0e-13


def test_adaptation_slows_later_interspike_intervals() -> None:
    """Threshold adaptation lengthens inter-spike intervals under constant drive."""
    n = AdaptiveThresholdIFNeuron()
    spike_steps: list[int] = []
    for index in range(5000):
        if n.step(100.0) == 1:
            spike_steps.append(index)
    assert len(spike_steps) >= 2
    intervals = np.diff(spike_steps)
    assert intervals[-1] >= intervals[0]


def test_varied_drive_event_vector_matches_candidate_crossing_rule() -> None:
    """Every emitted spike is a candidate crossing with the documented reset."""
    n = AdaptiveThresholdIFNeuron()
    drive = 16.0 + 8.0 * np.sin(np.arange(512, dtype=np.float64) * 0.041)
    v_prev, theta_prev = n.v, n.theta
    for value in drive:
        decay_v = math.exp(-n.dt / n.tau_m)
        decay_theta = math.exp(-n.dt / n.tau_theta)
        candidate_v = (n.v_rest + value) + (v_prev - (n.v_rest + value)) * decay_v
        candidate_theta = n.theta_rest + (theta_prev - n.theta_rest) * decay_theta
        spike = n.step(float(value))
        assert spike == int(candidate_v >= candidate_theta)
        if spike:
            assert n.v == n.v_reset
            assert abs(n.theta - (candidate_theta + n.delta_theta)) < 1.0e-12
        v_prev, theta_prev = n.v, n.theta


def test_long_run_is_finite_deterministic_and_bounded() -> None:
    """A 20k-step varied run stays finite, deterministic, and bounded."""
    drive = 15.0 + 7.5 * np.sin(np.arange(20_000, dtype=np.float64) * 0.007)
    first = AdaptiveThresholdIFNeuron()
    second = AdaptiveThresholdIFNeuron()
    trace_first = first.simulate(drive, backend="python")
    trace_second = second.simulate(drive, backend="python")
    assert np.isfinite(trace_first["v"]).all()
    assert np.isfinite(trace_first["theta"]).all()
    assert np.all(trace_first["v"] <= 20.0)
    assert np.all(trace_first["theta"] >= -51.0)
    np.testing.assert_array_equal(trace_first["v"], trace_second["v"])
    np.testing.assert_array_equal(trace_first["theta"], trace_second["theta"])
