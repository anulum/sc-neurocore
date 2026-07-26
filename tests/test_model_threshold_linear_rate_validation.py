# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Threshold-linear rate validation tests

"""Constructor, runtime, output, batch, and backend validation contracts."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.threshold_linear_rate import ThresholdLinearRateNeuron


@pytest.mark.parametrize("field", ["r", "theta", "gain"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_constructor_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ThresholdLinearRateNeuron(**{field: value})


@pytest.mark.parametrize(("field", "value"), [("r", -1.0), ("gain", -1.0)])
def test_rejects_negative_rate_or_gain(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ThresholdLinearRateNeuron(**{field: value})


@pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_current_without_mutation(current: float) -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert neuron.r == 0.25


@pytest.mark.parametrize("field", ["r", "theta", "gain"])
def test_rejects_corrupted_runtime_contract_without_mutation(field: str) -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    setattr(neuron, field, np.nan)
    with pytest.raises(ValueError, match=field):
        neuron.step(3.0)
    if field != "r":
        assert neuron.r == 0.25


def test_rejects_overflowing_output_without_mutation() -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, gain=1.0e308)
    with pytest.raises(ValueError, match="rate output"):
        neuron.step(1.0e308)
    assert neuron.r == 0.25


@pytest.mark.parametrize("n_steps", [-1, 1.5, True])
def test_rejects_invalid_batch_length_without_mutation(n_steps: object) -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25)
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 3.0, backend="python")
    assert neuron.r == 0.25


def test_rejects_unknown_backend_without_mutation() -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25)
    with pytest.raises(ValueError, match="backend must be"):
        neuron.simulate(1, 3.0, backend="cuda")
    assert neuron.r == 0.25
