# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC stochastic rate-adaptation isolation tests

"""Scalar behaviour tests for the retained stochastic adaptation model."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.sc_stochastic_rate_adaptation import (
    SCStochasticRateAdaptationNeuron,
)
from tests.sc_stochastic_rate_adaptation_support import rk4_reference


def test_construction() -> None:
    neuron = SCStochasticRateAdaptationNeuron()
    assert neuron.a == 0.0
    assert neuron.f_max == 200.0


def test_step_returns_binary() -> None:
    assert SCStochasticRateAdaptationNeuron(seed=1).step(10.0) in (0, 1)


def test_spikes_under_drive() -> None:
    neuron = SCStochasticRateAdaptationNeuron(seed=2)
    assert sum(neuron.step(50.0) for _ in range(10_000)) > 0


def test_adaptation_increases() -> None:
    neuron = SCStochasticRateAdaptationNeuron(seed=3)
    for _ in range(1_000):
        neuron.step(30.0)
    assert neuron.a > 0.0


def test_adaptation_candidate_matches_rk4_reference() -> None:
    neuron = SCStochasticRateAdaptationNeuron(a=0.35, dt=0.25, seed=5)
    expected_a, expected_p = rk4_reference(neuron, 12.5)
    candidate_a, candidate_p = neuron._rk4_candidate(12.5)

    assert candidate_a == pytest.approx(expected_a, rel=1e-14, abs=1e-14)
    assert candidate_p == pytest.approx(expected_p, rel=1e-14, abs=1e-14)


def test_step_commits_rk4_candidate_before_sampling() -> None:
    neuron = SCStochasticRateAdaptationNeuron(a=0.25, dt=0.5, seed=6)
    expected_a, _ = rk4_reference(neuron, 15.0)
    neuron.step(15.0)
    assert neuron.a == pytest.approx(expected_a, rel=1e-14, abs=1e-14)


def test_exponential_hazard_keeps_probability_bounded() -> None:
    neuron = SCStochasticRateAdaptationNeuron(f_max=1.0e6, dt=1.0, seed=7)
    _, probability = neuron._rk4_candidate(1.0e6)
    assert probability == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_seeded_sequences_are_reproducible() -> None:
    left = SCStochasticRateAdaptationNeuron(seed=123)
    right = SCStochasticRateAdaptationNeuron(seed=123)
    assert [left.step(25.0) for _ in range(50)] == [right.step(25.0) for _ in range(50)]


def test_f_onset_is_sigmoid() -> None:
    neuron = SCStochasticRateAdaptationNeuron()
    assert neuron._f_onset(50.0) > neuron._f_onset(0.0)


def test_state_remains_finite() -> None:
    neuron = SCStochasticRateAdaptationNeuron()
    for _ in range(5_000):
        neuron.step(100.0)
    assert np.isfinite(neuron.a)


def test_reset() -> None:
    neuron = SCStochasticRateAdaptationNeuron()
    for _ in range(100):
        neuron.step(30.0)
    neuron.reset()
    assert neuron.a == 0.0
