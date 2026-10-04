# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source EnergyLIF mutable state contracts

"""Exercise configuration edits, atomic reset and dynamic-state recovery."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron
from tests.energy_lif_contract_support import INVALID_CONFIGURATIONS, PARAMETERS


@pytest.mark.parametrize("parameters", INVALID_CONFIGURATIONS)
def test_source_constructor_refuses_invalid_configuration(parameters: dict[str, float]) -> None:
    """Refuse undefined normalization and configurations outside enrolled bounds."""
    with pytest.raises(ValueError):
        EnergyLIFNeuron(**parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"epsilon_0": 0.0},
        {"e_0": -201.0},
        {"alpha": 11.0},
        {"alpha": 1e-300, "epsilon_0": 1e-300},
        {"alpha": math.inf},
    ],
)
def test_mutated_configuration_reset_is_atomic(parameters: dict[str, float]) -> None:
    """An invalid configuration edit cannot commit either reset state."""
    neuron = EnergyLIFNeuron()
    assert neuron.step(80.0) == 0
    before = (neuron.v, neuron.epsilon)
    for name, value in parameters.items():
        setattr(neuron, name, value)
    with pytest.raises(ValueError):
        neuron.reset()
    assert (neuron.v, neuron.epsilon) == before
    neuron.e_0, neuron.alpha, neuron.epsilon_0 = -62.5, 1.0, 0.5
    neuron.reset()
    assert (neuron.v, neuron.epsilon) == (-62.5, 0.5)
    assert neuron.step(80.0) == 0


@pytest.mark.parametrize("e_0,alpha", [(-200.0, 10.0), (100.0, 0.5)])
def test_valid_reset_recovers_invalid_dynamic_state(e_0: float, alpha: float) -> None:
    """Inclusive reset bounds remain accepted even when dynamic state needs recovery."""
    neuron = EnergyLIFNeuron(e_0=e_0, alpha=alpha)
    neuron.v, neuron.epsilon = math.nan, -1.0
    neuron.reset()
    assert (neuron.v, neuron.epsilon) == (e_0, alpha * neuron.epsilon_0)


@pytest.mark.parametrize("field", [name for name, _, _ in PARAMETERS])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_source_nonfinite_configuration_is_refused(field: str, value: float) -> None:
    """Each source parameter rejects nonfinite configuration before any stepping."""
    with pytest.raises(ValueError):
        EnergyLIFNeuron(**{field: value})


@pytest.mark.parametrize("current", [math.nan, math.inf, -math.inf, 1e308, 1e6])
def test_source_transition_refusal_and_valid_retry(current: float) -> None:
    """Overflow and nonfinite currents produce authored refusal with atomic state."""
    neuron = EnergyLIFNeuron()
    control = EnergyLIFNeuron()
    before = (neuron.v, neuron.epsilon)
    with pytest.raises(ValueError):
        neuron.step(current)
    assert (neuron.v, neuron.epsilon) == before
    assert neuron.step(80.0) == control.step(80.0)
    assert (neuron.v, neuron.epsilon) == (control.v, control.epsilon)


def test_source_post_spike_energy_refusal_is_atomic() -> None:
    """An unaffordable event cannot commit either voltage reset or energy debt."""
    neuron = EnergyLIFNeuron(v=-58.8, delta=1.0)
    before = (neuron.v, neuron.epsilon)
    with pytest.raises(ValueError, match="post-spike energy"):
        neuron.step(0.0)
    assert (neuron.v, neuron.epsilon) == before
    neuron.delta = 0.01
    assert neuron.step(0.0) == 1
