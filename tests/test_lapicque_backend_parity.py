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
    SOURCE_TRACE_ATOL,
    TRACE_ATOL,
    configured,
    run_backend,
    source_configured,
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
def test_source_profile_complete_packet_matches_python(backend: str) -> None:
    """Carry the source circuit, latch, and every aligned event across each lane."""
    python = source_configured()
    native = source_configured()
    expected_voltage, expected_events = python.simulate_complete(4_000, 24.0, backend="python")
    voltage, events = native.simulate_complete(4_000, 24.0, backend=backend)
    np.testing.assert_allclose(voltage, expected_voltage, atol=SOURCE_TRACE_ATOL, rtol=0.0)
    np.testing.assert_array_equal(events, expected_events)
    assert int(events.sum()) == 1
    assert native.v == pytest.approx(python.v, abs=SOURCE_TRACE_ATOL)
    assert native.excited is python.excited is True


@pytest.mark.parametrize("backend", COMPILED_BACKENDS)
def test_empty_run_preserves_state(backend: str) -> None:
    """Return an empty trace without discarding the initial voltage."""
    neuron = LapicqueNeuron() if backend == "rust" else configured()
    before = neuron.v
    trace, spikes = neuron.simulate(0, 2.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert neuron.v == before


def test_rust_complete_binding_accepts_the_full_sc_contract() -> None:
    """Keep Rust on the same explicit full-parameter path as every native lane."""
    neuron = configured()
    expected = configured()
    trace, events = neuron.simulate_complete(300, 2.2, backend="rust")
    expected_trace, expected_events = expected.simulate_complete(300, 2.2, backend="python")
    np.testing.assert_allclose(trace, expected_trace, atol=TRACE_ATOL, rtol=0.0)
    np.testing.assert_array_equal(events, expected_events)
    assert neuron.v == pytest.approx(expected.v, abs=TRACE_ATOL)
