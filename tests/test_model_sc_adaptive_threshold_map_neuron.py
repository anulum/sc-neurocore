# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained SC adaptive-threshold-map contract

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models import SCAdaptiveThresholdMapNeuron as PublicNeuron
from sc_neurocore.neurons.models.sc_adaptive_threshold_map_neuron import (
    SCAdaptiveThresholdMapNeuron,
)


def test_public_identity_defaults_and_first_step() -> None:
    neuron = SCAdaptiveThresholdMapNeuron()
    assert PublicNeuron is SCAdaptiveThresholdMapNeuron
    assert neuron.step(0.6) == 1
    assert (neuron.x, neuron.theta) == (1.35, 0.0)


@pytest.mark.parametrize(
    "field",
    ["x", "theta", "k", "beta", "gamma", "theta_spike", "x_threshold"],
)
def test_nonfinite_configuration_is_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        SCAdaptiveThresholdMapNeuron(**{field: math.nan})


@pytest.mark.parametrize(
    "field,value",
    [
        ("x", 5.1),
        ("theta", -5.1),
        ("k", 5.1),
        ("beta", -0.1),
        ("gamma", 2.1),
        ("theta_spike", -0.1),
        ("x_threshold", 2.1),
    ],
)
def test_project_parameter_and_state_bounds_are_enforced(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        SCAdaptiveThresholdMapNeuron(**{field: value})


def test_non_numeric_values_and_nonfinite_candidates_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="x must be numeric"):
        SCAdaptiveThresholdMapNeuron(x=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="current must be numeric"):
        SCAdaptiveThresholdMapNeuron().step(object())  # type: ignore[arg-type]
    neuron = SCAdaptiveThresholdMapNeuron()
    monkeypatch.setattr(neuron, "_sigmoid", lambda _value: math.inf)
    with pytest.raises(FloatingPointError, match="candidate"):
        neuron.step(0.0)

    neuron = SCAdaptiveThresholdMapNeuron()
    monkeypatch.setattr(neuron, "_validate_configuration", lambda: None)
    neuron.gamma = math.inf
    with pytest.raises(FloatingPointError, match="candidate"):
        neuron.step(0.0)


def test_failure_is_atomic_and_clamps_extreme_drive() -> None:
    neuron = SCAdaptiveThresholdMapNeuron()
    before = (neuron.x, neuron.theta)
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.nan)
    assert (neuron.x, neuron.theta) == before
    neuron.step(1e300)
    assert neuron.x == 5.0


def test_reset_preserves_project_parameters() -> None:
    neuron = SCAdaptiveThresholdMapNeuron(k=2.0, beta=0.8, gamma=0.5)
    neuron.step(1.0)
    neuron.reset()
    assert (neuron.x, neuron.theta, neuron.k, neuron.beta, neuron.gamma) == (
        0.0,
        0.0,
        2.0,
        0.8,
        0.5,
    )


def test_batch_updates_owner_and_returns_complete_receipts() -> None:
    neuron = SCAdaptiveThresholdMapNeuron()
    result = neuron.simulate(np.full(32, 0.6), backend="python")
    assert set(result) == {"x", "theta", "spikes", "x_final", "theta_final", "spike_count"}
    assert neuron.x == result["x_final"]
    assert neuron.theta == result["theta_final"]
