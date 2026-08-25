# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC stochastic rate-adaptation validation tests

"""Failure-atomicity tests for the retained stochastic adaptation model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models.sc_stochastic_rate_adaptation import (
    SCStochasticRateAdaptationNeuron,
)


@pytest.mark.parametrize("a", [-1.0, np.nan, np.inf, -np.inf])
def test_rejects_negative_or_non_finite_adaptation_state(a: float) -> None:
    with pytest.raises(ValueError, match="a"):
        SCStochasticRateAdaptationNeuron(a=a)


@pytest.mark.parametrize("field", ["f_max", "beta", "tau_a", "dt"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_rejects_non_positive_or_non_finite_scale_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        SCStochasticRateAdaptationNeuron(**{field: value})


@pytest.mark.parametrize("field", ["i_half", "delta_a"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_threshold_and_adaptation_gain(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        SCStochasticRateAdaptationNeuron(**{field: value})


def test_rejects_negative_adaptation_gain() -> None:
    with pytest.raises(ValueError, match="delta_a"):
        SCStochasticRateAdaptationNeuron(delta_a=-1.0)


@pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_current_before_state_mutation(current: float) -> None:
    neuron = SCStochasticRateAdaptationNeuron(a=0.5)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert neuron.a == 0.5


def test_rejects_non_finite_adaptation_update_before_state_mutation() -> None:
    neuron = SCStochasticRateAdaptationNeuron(f_max=1.0e-306, delta_a=1.0e308, dt=1.0e308, a=0.5)
    with pytest.raises(ValueError, match="adaptation RK4"):
        neuron.step(100.0)
    assert neuron.a == 0.5


@pytest.mark.parametrize("seed", [np.nan, np.inf, -1, True, 2**64])
def test_rejects_invalid_seed(seed: Any) -> None:
    with pytest.raises(ValueError, match="seed"):
        SCStochasticRateAdaptationNeuron(seed=seed)
