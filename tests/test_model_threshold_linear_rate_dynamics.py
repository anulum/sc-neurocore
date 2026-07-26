# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Threshold-linear rate dynamics tests

"""Defaults, rectified branches, memorylessness, and reset contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.threshold_linear_rate import ThresholdLinearRateNeuron


def test_defaults_and_float_output() -> None:
    neuron = ThresholdLinearRateNeuron()
    assert (neuron.r, neuron.theta, neuron.gain) == (0.0, 0.0, 1.0)
    assert isinstance(neuron.step(1.0), float)


@pytest.mark.parametrize(
    ("current", "expected"),
    [(1.0, 0.0), (1.5, 0.0), (3.0, 3.0), (-4.0, 0.0)],
)
def test_piecewise_linear_branches(current: float, expected: float) -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    assert neuron.step(current) == expected
    assert neuron.r == expected


def test_output_is_memoryless() -> None:
    neuron = ThresholdLinearRateNeuron(theta=1.5, gain=2.0)
    assert neuron.step(3.0) == 3.0
    assert neuron.step(1.0) == 0.0


def test_reset_preserves_configuration() -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, theta=-0.4, gain=2.5)
    neuron.step(3.0)
    neuron.reset()
    assert (neuron.r, neuron.theta, neuron.gain) == (0.0, -0.4, 2.5)
