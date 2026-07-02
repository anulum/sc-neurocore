# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fisher-Posner LIF contract tests

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.quantum_cognition.fisher_posner import (
    HybridFisherPosnerLIF,
    HybridFisherPosnerLIFNeuron,
)
from sc_neurocore.quantum_cognition.spin_pool import SpinPoolMPS


@pytest.fixture
def spin_pool() -> SpinPoolMPS:
    """A small deterministic spin pool for focused neuron contract tests."""
    return SpinPoolMPS(n_sites=4)


@pytest.mark.parametrize("neuron_id", [True, 0.5, "0"])
def test_hybrid_lif_rejects_non_integer_neuron_ids(
    spin_pool: SpinPoolMPS,
    neuron_id: object,
) -> None:
    """Neuron identifiers must be real integer site indices, not aliases."""
    with pytest.raises(TypeError, match="neuron_id"):
        HybridFisherPosnerLIF(neuron_id, spin_pool)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"dt": float("nan")},
        {"v_rest": float("inf")},
        {"v_threshold": float("nan")},
        {"v_reset": float("-inf")},
        {"tau_m": 0.0},
        {"tau_m": float("inf")},
        {"atp_initial": -0.1},
        {"atp_initial": 1.1},
        {"atp_consumption": 0.0},
        {"atp_consumption": 1.1},
        {"atp_basal_regeneration": -0.1},
        {"atp_basal_regeneration": float("nan")},
    ],
)
def test_hybrid_lif_rejects_invalid_physical_parameters(
    spin_pool: SpinPoolMPS,
    kwargs: dict[str, float],
) -> None:
    """Constructor parameters must stay finite and inside physical domains."""
    with pytest.raises(ValueError):
        HybridFisherPosnerLIF(0, spin_pool, **kwargs)


@pytest.mark.parametrize("current", [float("nan"), float("inf"), float("-inf")])
def test_hybrid_lif_rejects_non_finite_step_current_without_mutation(
    spin_pool: SpinPoolMPS,
    current: float,
) -> None:
    """Invalid input current must fail before counters or state mutate."""
    neuron = HybridFisherPosnerLIF(0, spin_pool)
    before = neuron.get_state()

    with pytest.raises(ValueError, match="I_in"):
        neuron.step(current)

    assert neuron.get_state() == before


def test_hybrid_lif_rejects_non_finite_voltage_assignment(spin_pool: SpinPoolMPS) -> None:
    """Voltage assignment must not admit NaN or infinite membrane state."""
    neuron = HybridFisherPosnerLIF(0, spin_pool)

    with pytest.raises(ValueError, match="v"):
        neuron.v = float("nan")

    assert neuron.v == -70.0


def test_hybrid_lif_reset_alias_restores_resting_state(spin_pool: SpinPoolMPS) -> None:
    """The NeuronProtocol reset alias must delegate to the state reset path."""
    neuron = HybridFisherPosnerLIF(0, spin_pool)
    neuron.v = -55.0
    neuron.atp_level = 0.2

    neuron.reset()

    assert neuron.v == -70.0
    assert neuron.atp_level == 1.0
    assert neuron.get_state()["total_steps"] == 0


def test_population_wrapper_wraps_sites_and_exposes_contract_methods() -> None:
    """The population wrapper must expose stable state, reset, and repr paths."""
    HybridFisherPosnerLIFNeuron._reset_pools()

    first = HybridFisherPosnerLIFNeuron(n_sites=1)
    second = HybridFisherPosnerLIFNeuron(n_sites=1)
    second.v = -55.0
    second.reset_state()
    state: dict[str, Any] = second.get_state()

    assert first.get_state()["neuron_id"] == 0
    assert state["neuron_id"] == 0
    assert second.v_threshold == -50.0
    assert second.v_rest == -70.0
    assert second.v == -70.0
    assert "HybridFisherPosnerLIFNeuron" in repr(second)


def test_population_wrapper_rejects_non_finite_voltage_assignment() -> None:
    """The wrapper voltage alias must enforce the inner neuron's finite state."""
    HybridFisherPosnerLIFNeuron._reset_pools()
    neuron = HybridFisherPosnerLIFNeuron()

    with pytest.raises(ValueError, match="v"):
        neuron.v = float("inf")

    assert neuron.v == -70.0
