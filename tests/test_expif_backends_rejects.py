# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rejects) from former test_expif_backends.py

from __future__ import annotations

from tests.expif_backends_support import *  # noqa: F403


@pytest.mark.skipif(not expif._HAS_RUST, reason="current ExpIF batch binding is unavailable")
def test_rust_accepts_complete_non_default_contract() -> None:
    """Exercise the full-parameter production batch rather than a default-only shim."""
    neuron = ExpIFNeuron(v=-60.0)
    voltage, refractory, events = neuron.simulate_complete(4, 0.0, backend="rust")
    assert voltage.shape == refractory.shape == events.shape == (4,)
    assert neuron.v == voltage[-1]
    assert neuron.refractory_remaining == refractory[-1]


@pytest.mark.parametrize("n_steps", [-1, 1.0, True])
def test_invalid_step_count_fails_before_mutation(n_steps: object) -> None:
    """Reject negative and non-integer step counts at the public boundary."""
    neuron = ExpIFNeuron()
    before = (neuron.v, neuron.refractory_remaining)
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 0.0)
    assert (neuron.v, neuron.refractory_remaining) == before


def test_invalid_backend_fails_before_mutation() -> None:
    """Reject unknown dispatch selectors instead of silently using Python."""
    neuron = ExpIFNeuron()
    before = (neuron.v, neuron.refractory_remaining)
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 0.0, backend="cuda")
    assert (neuron.v, neuron.refractory_remaining) == before


def test_simulate_rejects_non_finite_current_before_mutation() -> None:
    """Apply the same finite-input boundary to every dispatcher path."""
    neuron = ExpIFNeuron()
    before = (neuron.v, neuron.refractory_remaining)
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, math.nan, backend="auto")
    assert (neuron.v, neuron.refractory_remaining) == before
