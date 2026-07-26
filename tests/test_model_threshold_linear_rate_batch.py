# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Threshold-linear rate batch tests

"""Scalar parity, empty-batch, determinism, and population contracts."""

from __future__ import annotations

import numpy as np

from sc_neurocore.network.population import Population
from sc_neurocore.neurons.models.threshold_linear_rate import ThresholdLinearRateNeuron


def test_python_batch_matches_scalar_steps() -> None:
    scalar = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    expected = np.asarray([scalar.step(3.0) for _ in range(32)])
    batched = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    actual = batched.simulate(32, 3.0, backend="python")
    np.testing.assert_array_equal(actual, expected)
    assert batched.r == scalar.r


def test_empty_batch_preserves_cached_output() -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    np.testing.assert_array_equal(neuron.simulate(0, 3.0, backend="python"), np.empty(0))
    assert neuron.r == 0.25


def test_deterministic_and_population_compatible() -> None:
    first = ThresholdLinearRateNeuron(theta=1.0, gain=2.0).simulate(100, 3.0, backend="python")
    second = ThresholdLinearRateNeuron(theta=1.0, gain=2.0).simulate(100, 3.0, backend="python")
    np.testing.assert_array_equal(first, second)
    assert Population(ThresholdLinearRateNeuron, n=10, label="tlr").n == 10
