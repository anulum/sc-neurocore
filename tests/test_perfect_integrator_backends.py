# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable Perfect Integrator polyglot backend parity

"""End-to-end parity and rejection contracts for every native lane."""

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import perfect_integrator as backends
from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron

_GOLDENS = (
    (0.0, 0),
    (0.333, 32),
    (0.7, 66),
    (2.0, 200),
    (3.0, 250),
    (5.0, 500),
    (20.0, 1000),
)
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], PerfectIntegratorNeuron] = PerfectIntegratorNeuron,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run one backend and return its trace, event count, and final state."""
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, neuron.v


def _configured() -> PerfectIntegratorNeuron:
    """Return a non-default state that exercises the complete native ABI."""
    return PerfectIntegratorNeuron(
        v=0.25,
        c_m=1.7,
        v_threshold=1.3,
        v_reset=-0.2,
        dt=0.37,
    )


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four real compiled lanes without a skipped parity surface."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize(("current", "expected_spikes"), _GOLDENS)
@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_compiled_backends_match_python_golden(
    backend: str,
    current: float,
    expected_spikes: int,
) -> None:
    """Preserve the complete bit-exact trace and source-bound events."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)
    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    np.testing.assert_array_equal(trace, reference_trace)
    assert state == reference_state


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_full_parameter_contract_matches_python(backend: str) -> None:
    """Carry every maintained numeric field across each full-parameter ABI."""
    reference_trace, reference_spikes, reference_state = _run(
        "python", current=2.2, n_steps=300, factory=_configured
    )
    trace, spikes, state = _run(backend, current=2.2, n_steps=300, factory=_configured)
    assert spikes == reference_spikes == 75
    np.testing.assert_array_equal(trace, reference_trace)
    assert state == reference_state


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_empty_run_preserves_state(backend: str) -> None:
    """Return an empty trace without discarding the initial voltage."""
    neuron = PerfectIntegratorNeuron() if backend == "rust" else _configured()
    before = neuron.v
    trace, spikes = neuron.simulate(0, 2.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert neuron.v == before


def test_rust_rejects_non_default_contract() -> None:
    """Keep the Rust engine class's factory-only parameter boundary explicit."""
    neuron = _configured()
    before = neuron.v
    with pytest.raises(RuntimeError, match="factory-default"):
        neuron.simulate(1, 0.0, backend="rust")
    assert neuron.v == before


@pytest.mark.parametrize("n_steps", [-1, 1.0, True])
def test_invalid_step_count_fails_before_mutation(n_steps: object) -> None:
    """Reject negative and non-integer step counts at the public boundary."""
    neuron = PerfectIntegratorNeuron()
    before = neuron.v
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 0.0)
    assert neuron.v == before


def test_invalid_backend_fails_before_mutation() -> None:
    """Reject unknown dispatch selectors instead of silently using Python."""
    neuron = PerfectIntegratorNeuron()
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 0.0, backend="cuda")
    assert neuron.v == 0.0


def test_non_finite_current_fails_before_mutation() -> None:
    """Apply the finite-input boundary to every dispatcher path."""
    neuron = PerfectIntegratorNeuron(v=0.25)
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, math.nan, backend="auto")
    assert neuron.v == 0.25


def test_auto_prefers_measured_first_mojo() -> None:
    """Route a non-default instance through measured-first Mojo."""
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="mojo")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_through_to_julia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Julia when the measured-first Mojo lane is unavailable."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="julia")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_through_to_go(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue to Go when Mojo and Julia are unavailable."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="go")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_through_to_factory_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Rust when every full-parameter compiled lane is unavailable."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    auto = PerfectIntegratorNeuron()
    expected = PerfectIntegratorNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="rust")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the Python floor when no compatible compiled lane is available."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    auto = PerfectIntegratorNeuron()
    expected = PerfectIntegratorNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="python")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def _c_arguments(neuron: PerfectIntegratorNeuron) -> tuple[float, ...]:
    """Return numeric fields in the C-ABI declaration order."""
    return (neuron.v, neuron.c_m, neuron.v_threshold, neuron.v_reset, neuron.dt)


@pytest.mark.parametrize("backend", ("go", "mojo"))
@pytest.mark.parametrize("current", (math.nan, 1.0e308))
def test_c_abi_rejects_invalid_run_without_writing_output(
    backend: str,
    current: float,
) -> None:
    """Reject invalid work before emitting any caller-visible row."""
    neuron = PerfectIntegratorNeuron(
        v=0.25,
        v_threshold=1.0e308,
        c_m=1.0e-308 if math.isfinite(current) else 1.0,
    )
    output = np.full(2, -999.0, dtype=np.float64)
    if backend == "go":
        assert backends._go_lib is not None
        result = backends._go_lib.perfect_integrator_simulate_c(
            *_c_arguments(neuron),
            1,
            current,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        assert backends._mojo_lib is not None
        result = backends._mojo_lib.perfect_integrator_simulate_c(
            *_c_arguments(neuron), 1, current, int(output.ctypes.data)
        )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(2, -999.0, dtype=np.float64))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate a native non-finite candidate into mutation-free failure."""
    neuron = PerfectIntegratorNeuron(v=0.25, v_threshold=1.0e308, c_m=1.0e-308)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, 1.0e308, backend=backend)
    assert neuron.v == 0.25


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable failure instead of silently falling back."""
    monkeypatch.setattr(backends, f"ensure_{backend}_loaded", lambda: False)
    with pytest.raises(RuntimeError, match=backend.title()):
        PerfectIntegratorNeuron().simulate(1, 0.0, backend=backend)


def test_requested_rust_backend_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep explicit Rust requests fail-closed when the engine is absent."""
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    monkeypatch.setattr(backends, "_EnginePerfectIntegratorCls", None)
    with pytest.raises(RuntimeError, match="Rust PerfectIntegrator backend"):
        PerfectIntegratorNeuron().simulate(1, 0.0, backend="rust")
