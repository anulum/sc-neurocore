# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mutable normalized energy-LIF reset and state contracts

"""Exercise real source constructors, edited configuration and atomic reset."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from sc_neurocore.neurons.models.sc_normalized_energy_lif import SCNormalizedEnergyLIFNeuron
from tests.test_sc_normalized_energy_lif_engine_binding_configuration import (
    FIELDS,
    INVALID_CONFIGURATIONS,
    INVALID_REST_CONFIGURATIONS,
)


@pytest.mark.parametrize("overrides", [*INVALID_CONFIGURATIONS, *INVALID_REST_CONFIGURATIONS])
def test_source_constructor_refuses_invalid_configuration(overrides: dict[str, float]) -> None:
    """Every finite invalid domain refuses before any temporal state exists."""
    with pytest.raises(ValueError):
        SCNormalizedEnergyLIFNeuron(**overrides)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_source_configuration_refuses(field: str, value: float) -> None:
    """The source constructor rejects all nonfinite parameter fields."""
    with pytest.raises(ValueError):
        SCNormalizedEnergyLIFNeuron(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        *INVALID_REST_CONFIGURATIONS,
        {"tau_m": 0.0},
        {"epsilon_0": -1.0},
        {"alpha": -1.0},
        {"v_reset": 101.0},
        {"v_threshold": -70.0},
        {"v_rest": math.inf},
    ],
)
def test_mutated_configuration_reset_refusal_is_atomic(overrides: dict[str, float]) -> None:
    """An unsafe configuration edit cannot commit either resting value."""
    cell = SCNormalizedEnergyLIFNeuron()
    cell.step(30.0)
    before = (cell.v, cell.epsilon)
    for field, value in overrides.items():
        setattr(cell, field, value)
    with pytest.raises(ValueError):
        cell.reset()
    assert (cell.v, cell.epsilon) == before
    for field in overrides:
        setattr(cell, field, getattr(SCNormalizedEnergyLIFNeuron(), field))
    cell.reset()
    assert (cell.v, cell.epsilon) == (-70.0, 1.0)
    assert cell.step(30.0) == 0


@pytest.mark.parametrize("rest", [-200.0, 100.0])
@pytest.mark.parametrize("energy", [0.0, 1.0])
def test_valid_reset_recovers_bad_dynamic_state(rest: float, energy: float) -> None:
    """Inclusive reset endpoints recover independently invalid dynamic fields."""
    cell = SCNormalizedEnergyLIFNeuron(
        v_rest=rest, v_threshold=101.0, epsilon=energy, epsilon_0=energy
    )
    cell.v, cell.epsilon = math.nan, -1.0
    cell.reset()
    assert (cell.v, cell.epsilon) == (rest, energy)
    assert cell.step(0.0) == 0 and (cell.v, cell.epsilon) == (rest, energy)


@pytest.mark.parametrize("current", [math.nan, math.inf, -math.inf, 1e308, 1e6])
def test_transition_refusal_preserves_state_and_retry(current: float) -> None:
    """Nonfinite and unsafe candidates never commit partial source state."""
    cell = SCNormalizedEnergyLIFNeuron()
    control = replace(cell)
    before = (cell.v, cell.epsilon)
    with pytest.raises(ValueError):
        cell.step(current)
    assert (cell.v, cell.epsilon) == before
    assert cell.step(30.0) == control.step(30.0)
    assert (cell.v, cell.epsilon) == (control.v, control.epsilon)
