# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Nagumo–Sato hand-model contract

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models import NagumoSatoMapNeuron as PublicNeuron
from sc_neurocore.neurons.models.nagumo_sato_map_neuron import NagumoSatoMapNeuron


def test_public_identity_and_source_defaults() -> None:
    neuron = NagumoSatoMapNeuron()
    assert PublicNeuron is NagumoSatoMapNeuron
    assert (neuron.y, neuron.k, neuron.alpha, neuron.bias) == (0.1, 0.6, 1.0, 0.2)
    assert neuron.x == neuron.output() == 1


def test_first_steps_match_source_equation_and_h_zero_is_one() -> None:
    neuron = NagumoSatoMapNeuron()
    expected = [(-0.74, 0), (-0.244, 0), (0.0536, 1)]
    for y, event in expected:
        assert neuron.step() == event
        assert neuron.y == pytest.approx(y, abs=1e-15)
    zero = NagumoSatoMapNeuron(y=0.0)
    assert zero.output() == 1


@pytest.mark.parametrize("field", ["y", "k", "alpha", "bias"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_configuration_is_rejected(field: str, value: float) -> None:
    error = FloatingPointError if field == "y" else ValueError
    with pytest.raises(error):
        NagumoSatoMapNeuron(**{field: value})


@pytest.mark.parametrize("kwargs", [{"k": -0.1}, {"k": 1.0}, {"alpha": 0.0}])
def test_source_parameter_bounds_are_enforced(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        NagumoSatoMapNeuron(**kwargs)


def test_non_numeric_values_and_overflowing_candidate_are_rejected() -> None:
    with pytest.raises(ValueError, match="y must be numeric"):
        NagumoSatoMapNeuron(y=object())  # type: ignore[arg-type]
    neuron = NagumoSatoMapNeuron()
    neuron.y = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="internal state must be numeric"):
        neuron.output()
    neuron = NagumoSatoMapNeuron()
    neuron.k = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="parameters must be numeric"):
        neuron.step()
    with pytest.raises(ValueError, match="current must be numeric"):
        NagumoSatoMapNeuron().step(object())  # type: ignore[arg-type]
    with pytest.raises(FloatingPointError, match="candidate"):
        NagumoSatoMapNeuron(bias=1e308).step(1e308)


def test_failure_is_atomic_and_reset_preserves_parameters() -> None:
    neuron = NagumoSatoMapNeuron(k=0.5, alpha=2.0, bias=0.7)
    before = neuron.y
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.nan)
    assert neuron.y == before
    neuron.step(0.1)
    neuron.reset()
    assert (neuron.y, neuron.k, neuron.alpha, neuron.bias) == (0.1, 0.5, 2.0, 0.7)


def test_batch_updates_owner_and_returns_complete_receipts() -> None:
    neuron = NagumoSatoMapNeuron()
    result = neuron.simulate(np.zeros(12), backend="python")
    assert set(result) == {"y", "x", "spikes", "y_final", "x_final", "spike_count"}
    np.testing.assert_array_equal(result["x"], result["spikes"])
    assert neuron.y == result["y_final"]
