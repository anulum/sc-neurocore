# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable Quadratic IF polyglot backend parity

"""End-to-end parity and rejection contracts for every native QIF lane."""

from __future__ import annotations

import ctypes
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import quadratic_if as backends
from sc_neurocore.neurons.models.quadratic_if import QuadraticIFNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_GOLDENS = (
    (0.0, 0),
    (0.333, 2),
    (0.5, 3),
    (1.0, 6),
    (2.0, 11),
    (5.0, 26),
    (20.0, 100),
    (50.0, 250),
)
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_TRACE_ATOL = 2.0e-12


def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], QuadraticIFNeuron] = QuadraticIFNeuron,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run one backend and return its trace, event count, and final state."""
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, neuron.v


def _configured() -> QuadraticIFNeuron:
    """Return a non-default state exercising the complete native ABI."""
    return QuadraticIFNeuron(v=-0.37, v_reset=-1.3, v_peak=1.7, dt=0.037)


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
    """Preserve exact events and a tight cross-libm voltage envelope."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)
    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    np.testing.assert_allclose(trace, reference_trace, rtol=0.0, atol=_TRACE_ATOL)
    assert state == pytest.approx(reference_state, rel=0.0, abs=_TRACE_ATOL)


def test_rust_safety_executable_matches_python_trace() -> None:
    """Run the actual accel/rust/safety QIF module against the Python trace."""
    command = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--manifest-path",
        "src/sc_neurocore/accel/rust/Cargo.toml",
        "--example",
        "quadratic_if_trace",
        "--",
        "-1.0",
        "-1.0",
        "1.0",
        "0.01",
        "400",
        "0.5",
    ]
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    rows = [line.split() for line in completed.stdout.splitlines() if line.startswith("QIF_TRACE ")]
    assert len(rows) == 400
    rust_events = [int(row[1]) for row in rows]
    rust_trace = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    python_trace, python_spikes, _ = _run("python", current=0.5, n_steps=400)
    assert sum(rust_events) == python_spikes
    np.testing.assert_allclose(rust_trace, python_trace, rtol=0.0, atol=_TRACE_ATOL)


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_full_parameter_contract_matches_python(backend: str) -> None:
    """Carry every maintained numeric field across each full-parameter ABI."""
    reference_trace, reference_spikes, reference_state = _run(
        "python", current=2.2, n_steps=400, factory=_configured
    )
    trace, spikes, state = _run(backend, current=2.2, n_steps=400, factory=_configured)
    assert reference_spikes > 0
    assert spikes == reference_spikes
    np.testing.assert_allclose(trace, reference_trace, rtol=0.0, atol=_TRACE_ATOL)
    assert state == pytest.approx(reference_state, rel=0.0, abs=_TRACE_ATOL)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_empty_run_preserves_state(backend: str) -> None:
    """Return an empty trace without discarding the initial voltage."""
    neuron = QuadraticIFNeuron() if backend == "rust" else _configured()
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
    neuron = QuadraticIFNeuron()
    before = neuron.v
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 0.0)
    assert neuron.v == before


def test_invalid_backend_fails_before_mutation() -> None:
    """Reject unknown dispatch selectors instead of silently using Python."""
    neuron = QuadraticIFNeuron()
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 0.0, backend="cuda")
    assert neuron.v == -1.0


def test_non_finite_current_fails_before_mutation() -> None:
    """Apply the finite-input boundary to every dispatcher path."""
    neuron = QuadraticIFNeuron(v=-0.25)
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, math.nan, backend="auto")
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.nan)
    assert neuron.v == -0.25


@pytest.mark.parametrize(
    "parameters",
    (
        {"v": math.nan},
        {"v": 1.0},
        {"v_reset": 1.0},
        {"dt": 0.0},
    ),
)
def test_invalid_model_contract_is_rejected_at_construction(
    parameters: dict[str, float],
) -> None:
    """Reject each invalid maintained field before a neuron can execute."""
    with pytest.raises(ValueError):
        QuadraticIFNeuron(**parameters)


def test_negative_current_exact_flow_covers_stable_and_reset_regimes() -> None:
    """Preserve the Riccati fixed point, regular decay, and finite-time reset."""
    fixed = QuadraticIFNeuron(v=-1.0)
    assert fixed.step(-1.0) == 0
    assert fixed.v == -1.0

    decaying = QuadraticIFNeuron(v=-0.25)
    assert decaying.step(-1.0) == 0
    assert math.isfinite(decaying.v)

    crossing = QuadraticIFNeuron(v=1.1, v_peak=3.0, dt=20.0)
    assert crossing.step(-1.0) == 1
    assert crossing.v == crossing.v_reset

    zero_current_crossing = QuadraticIFNeuron(v=0.5, v_peak=2.0, dt=3.0)
    assert zero_current_crossing.step(0.0) == 1
    assert zero_current_crossing.v == zero_current_crossing.v_reset


def test_exact_flow_rejects_overflow_without_mutation() -> None:
    """Translate a non-finite analytic candidate into mutation-free failure."""
    neuron = QuadraticIFNeuron(v=-0.25)
    before = neuron.v
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(-1.0e308)
    assert neuron.v == before


def test_reset_restores_only_the_runtime_state() -> None:
    """Restore voltage while retaining the configured parameters."""
    neuron = _configured()
    expected = (neuron.v_reset, neuron.v_peak, neuron.dt)
    neuron.step(2.2)
    neuron.reset()
    assert (neuron.v, neuron.v_peak, neuron.dt) == expected


def test_auto_prefers_go_without_initialising_julia() -> None:
    """Route through Go without paying Julia runtime initialisation."""
    with patch.object(backends, "ensure_julia_loaded") as ensure_julia:
        auto = _configured()
        expected = _configured()
        auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="go")
    ensure_julia.assert_not_called()
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_through_to_julia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Julia when the Go shared library is unavailable."""
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="julia")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_through_to_mojo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue to Mojo when Go and Julia are unavailable."""
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="mojo")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_through_to_factory_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Rust when every full-parameter compiled lane is unavailable."""
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    auto = QuadraticIFNeuron()
    expected = QuadraticIFNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="rust")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the Python floor when no compatible compiled lane is available."""
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    auto = QuadraticIFNeuron()
    expected = QuadraticIFNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="python")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def _c_arguments(neuron: QuadraticIFNeuron) -> tuple[float, ...]:
    """Return numeric fields in the C-ABI declaration order."""
    return (neuron.v, neuron.v_reset, neuron.v_peak, neuron.dt)


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejects_invalid_run_without_writing_output(backend: str) -> None:
    """Reject invalid work before emitting any caller-visible row."""
    neuron = QuadraticIFNeuron(v=-0.25)
    output = np.full(2, -999.0, dtype=np.float64)
    if backend == "go":
        assert backends._go_lib is not None
        result = backends._go_lib.quadratic_if_simulate_c(
            *_c_arguments(neuron),
            1,
            math.nan,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        assert backends._mojo_lib is not None
        result = backends._mojo_lib.quadratic_if_simulate_c(
            *_c_arguments(neuron), 1, math.nan, int(output.ctypes.data)
        )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(2, -999.0, dtype=np.float64))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate a native non-finite candidate into mutation-free failure."""
    neuron = QuadraticIFNeuron(v=-0.25)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, -1.0e308, backend=backend)
    assert neuron.v == -0.25


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable failure instead of silently falling back."""
    monkeypatch.setattr(backends, f"ensure_{backend}_loaded", lambda: False)
    with pytest.raises(RuntimeError, match=backend.title()):
        QuadraticIFNeuron().simulate(1, 0.0, backend=backend)


def test_requested_rust_backend_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep explicit Rust requests fail-closed when the engine is absent."""
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    monkeypatch.setattr(backends, "_EngineQuadraticIFCls", None)
    with pytest.raises(RuntimeError, match="Rust QuadraticIF backend"):
        QuadraticIFNeuron().simulate(1, 0.0, backend="rust")
