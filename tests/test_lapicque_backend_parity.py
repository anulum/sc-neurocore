# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque compiled-backend parity contracts

from __future__ import annotations


import numpy as np
import pytest

from sc_neurocore.accel import lapicque as backends
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from tests.lapicque_backends_support import (
    COMPILED_BACKENDS,
    GOLDENS,
    TRACE_ATOL,
    configured,
    run_backend,
)


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four real compiled lanes without a skipped parity surface."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize(("current", "expected_spikes"), GOLDENS)
@pytest.mark.parametrize("backend", COMPILED_BACKENDS)
def test_compiled_backends_match_python_golden(
    backend: str,
    current: float,
    expected_spikes: int,
) -> None:
    """Preserve the complete trace, final state, and source-bound events."""
    reference_trace, reference_spikes, reference_state = run_backend("python", current=current)
    trace, spikes, state = run_backend(backend, current=current)
    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    np.testing.assert_allclose(trace, reference_trace, atol=TRACE_ATOL, rtol=0.0)
    assert state == pytest.approx(reference_state, abs=TRACE_ATOL)


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_full_parameter_contract_matches_python(backend: str) -> None:
    """Carry every maintained numeric field across each full-parameter ABI."""
    reference_trace, reference_spikes, reference_state = run_backend(
        "python", current=2.2, n_steps=300, factory=configured
    )
    trace, spikes, state = run_backend(backend, current=2.2, n_steps=300, factory=configured)
    assert spikes == reference_spikes == 27
    np.testing.assert_allclose(trace, reference_trace, atol=TRACE_ATOL, rtol=0.0)
    assert state == pytest.approx(reference_state, abs=TRACE_ATOL)


@pytest.mark.parametrize("backend", COMPILED_BACKENDS)
def test_empty_run_preserves_state(backend: str) -> None:
    """Return an empty trace without discarding the initial voltage."""
    neuron = LapicqueNeuron() if backend == "rust" else configured()
    before = neuron.v
    trace, spikes = neuron.simulate(0, 2.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert neuron.v == before


def test_rust_rejects_non_default_contract() -> None:
    """Keep the Rust engine class's factory-only parameter boundary explicit."""
    neuron = configured()
    before = neuron.v
    with pytest.raises(RuntimeError, match="factory-default"):
        neuron.simulate(1, 0.0, backend="rust")
    assert neuron.v == before
