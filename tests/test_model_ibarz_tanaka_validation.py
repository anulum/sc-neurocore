# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka validation and reset tests

"""Parameter, request, atomic-failure, and reset contracts."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron


@pytest.mark.parametrize("overrides", ({"alpha": 0.0}, {"mu": 0.0}, {"sigma": math.inf}))
def test_invalid_parameter_topology_is_rejected(overrides: dict[str, float]) -> None:
    """Invalid source parameters cannot enter the runtime."""
    with pytest.raises(ValueError):
        IbarzTanakaMapNeuron(**overrides)


def test_failed_step_preserves_state() -> None:
    """Non-finite input fails before state mutation."""
    neuron = IbarzTanakaMapNeuron()
    before = (neuron.v, neuron.u)
    with pytest.raises(ValueError, match="current must be finite"):
        neuron.step(float("nan"))
    assert (neuron.v, neuron.u) == before


def test_failed_batch_preserves_state() -> None:
    """A mutable parameter fault rejects the batch without state mutation."""
    neuron = IbarzTanakaMapNeuron()
    before = (neuron.v, neuron.u)
    neuron.alpha = float("inf")
    with pytest.raises(ValueError, match="parameters must be finite"):
        neuron.simulate(10, 0.2, backend="python")
    assert (neuron.v, neuron.u) == before


def test_request_validation() -> None:
    """Batch bounds and backend selection are explicit."""
    neuron = IbarzTanakaMapNeuron()
    with pytest.raises(ValueError, match="n_steps must be an integer"):
        neuron.simulate(True)
    with pytest.raises(ValueError, match="n_steps must be between"):
        neuron.simulate(-1)
    with pytest.raises(ValueError, match="unsupported backend"):
        neuron.simulate(1, backend="cuda")


def test_reset_restores_only_the_source_initial_state() -> None:
    """Reset preserves parameters while restoring the published state placement."""
    neuron = IbarzTanakaMapNeuron(alpha=0.95)
    neuron.simulate(100, 0.2, backend="python")
    neuron.reset()
    assert (neuron.v, neuron.u) == (-1.0, -0.1)
    assert neuron.alpha == 0.95
