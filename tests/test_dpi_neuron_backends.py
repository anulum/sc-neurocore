# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable DPI polyglot backend parity

"""End-to-end parity and rejection contracts for every native DPI lane."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import dpi_neuron as backends
from sc_neurocore.neurons.models.dpi_neuron import DPINeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_GOLDENS = (
    (-0.1, 0),
    (0.0, 0),
    (1.0, 0),
    (2.0, 0),
    (3.0, 1),
    (5.0, 3),
    (10.0, 6),
    (20.0, 11),
    (50.0, 21),
)
_FULL_CONTRACT_BACKENDS = ("julia", "go", "mojo")
_COMPILED_BACKENDS = ("rust", *_FULL_CONTRACT_BACKENDS)
_STATE_ATOL = 5.0e-13


def _configured() -> DPINeuron:
    """Return a stable non-default state exercising the complete native ABI."""
    return DPINeuron(
        i_mem=0.37,
        i_ahp=0.08,
        refractory_time=0.0,
        i_threshold=1.3,
        i_reset=0.2,
        i_rest=0.15,
        i_tau=0.9,
        i_g=1.4,
        i_tau_ahp=0.12,
        i_ga=0.8,
        i_spike=4.2,
        i_0=0.02,
        kappa=0.65,
        alpha=8.0,
        tau=7.0,
        tau_ahp=45.0,
        refractory_period=0.6,
        dt=0.05,
    )


def _factory_values() -> tuple[float, ...]:
    """Return the 18-double native ABI prefix in public-model order."""
    neuron = DPINeuron()
    return (
        neuron.i_mem,
        neuron.i_ahp,
        neuron.refractory_time,
        neuron.i_threshold,
        neuron.i_reset,
        neuron.i_rest,
        neuron.i_tau,
        neuron.i_g,
        neuron.i_tau_ahp,
        neuron.i_ga,
        neuron.i_spike,
        neuron.i_0,
        neuron.kappa,
        neuron.alpha,
        neuron.tau,
        neuron.tau_ahp,
        neuron.refractory_period,
        neuron.dt,
    )


def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    configured: bool = False,
) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float]]:
    """Run one backend and return its trace, events, and all final states."""
    neuron = _configured() if configured else DPINeuron()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)


def _assert_state_parity(actual: npt.ArrayLike, expected: npt.ArrayLike) -> None:
    """Enforce the measured cross-runtime floating-point envelope."""
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(actual_array, expected_array, rtol=0.0, atol=_STATE_ATOL)


def _invoke_full_contract(runner: Callable[..., object]) -> object:
    """Invoke one configurable native runner with the public 20-field ABI."""
    return runner(*_factory_values(), 1, 0.0)


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four compiled lanes without a skipped parity surface."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize(("current", "expected_spikes"), _GOLDENS)
@pytest.mark.parametrize("backend", _FULL_CONTRACT_BACKENDS)
def test_full_contract_backends_match_python_factory_vector(
    backend: str,
    current: float,
    expected_spikes: int,
) -> None:
    """Preserve events, three states, and physical-domain handling."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)
    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    _assert_state_parity(trace, reference_trace)
    _assert_state_parity(state, reference_state)


@pytest.mark.parametrize(("current", "expected_spikes"), _GOLDENS)
def test_factory_rust_matches_python(current: float, expected_spikes: int) -> None:
    """Prove the fixed-constructor PyO3 engine executes the same recurrence."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run("rust", current=current)
    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    _assert_state_parity(trace, reference_trace)
    _assert_state_parity(state, reference_state)


def test_rust_safety_executable_matches_configured_python_trace() -> None:
    """Run the separately maintained Rust-safety module on all 18 fields."""
    command = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        "src/sc_neurocore/accel/rust/Cargo.toml",
        "--example",
        "dpi_neuron_trace",
        "--",
        "0.37",
        "0.08",
        "0.0",
        "1.3",
        "0.2",
        "0.15",
        "0.9",
        "1.4",
        "0.12",
        "0.8",
        "4.2",
        "0.02",
        "0.65",
        "8.0",
        "7.0",
        "45.0",
        "0.6",
        "0.05",
        "400",
        "5.0",
    ]
    environment = dict(os.environ)
    environment["CARGO_TARGET_DIR"] = str(_REPOSITORY / "target")
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=True,
    )
    rows = [line.split() for line in completed.stdout.splitlines() if line.startswith("DPI_TRACE ")]
    assert len(rows) == 400
    rust_events = [int(row[1]) for row in rows]
    rust_states = np.asarray([[float(value) for value in row[2:5]] for row in rows])
    python_trace, python_spikes, python_state = _run(
        "python",
        current=5.0,
        n_steps=400,
        configured=True,
    )
    assert sum(rust_events) == python_spikes == 4
    _assert_state_parity(rust_states[:, 0], python_trace)
    _assert_state_parity(rust_states[-1], python_state)


@pytest.mark.parametrize("backend", _FULL_CONTRACT_BACKENDS)
def test_complete_configured_contract_matches_python(backend: str) -> None:
    """Carry every state and parameter field through the native ABI."""
    reference_trace, reference_spikes, reference_state = _run(
        "python", current=5.0, n_steps=400, configured=True
    )
    trace, spikes, state = _run(backend, current=5.0, n_steps=400, configured=True)
    assert spikes == reference_spikes == 4
    _assert_state_parity(trace, reference_trace)
    _assert_state_parity(state, reference_state)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_empty_run_preserves_all_states(backend: str) -> None:
    """Return an empty trace without discarding any dynamic state."""
    neuron = DPINeuron() if backend == "rust" else _configured()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    trace, spikes = neuron.simulate(0, 5.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_rust_rejects_non_default_contract_without_mutation() -> None:
    """Fail closed outside the engine's fixed-constructor compatibility boundary."""
    neuron = _configured()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(RuntimeError, match="factory-default"):
        neuron.simulate(1, 0.0, backend="rust")
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_auto_prefers_go_without_initialising_other_runtimes() -> None:
    """Route through Go without initialising Julia or probing Mojo."""
    with (
        patch.object(backends, "ensure_julia_loaded") as ensure_julia,
        patch.object(backends, "ensure_mojo_loaded") as ensure_mojo,
    ):
        auto = _configured()
        expected = _configured()
        actual_trace, actual_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="go")
    ensure_julia.assert_not_called()
    ensure_mojo.assert_not_called()
    np.testing.assert_array_equal(actual_trace, expected_trace)
    assert (actual_spikes, auto.i_mem, auto.i_ahp) == (
        expected_spikes,
        expected.i_mem,
        expected.i_ahp,
    )


def test_auto_falls_through_julia_mojo_rust_and_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete documented fallback chain."""
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    julia_auto, julia_expected = _configured(), _configured()
    actual, events = julia_auto.simulate(100, 5.0)
    expected, expected_events = julia_expected.simulate(100, 5.0, backend="julia")
    np.testing.assert_array_equal(actual, expected)
    assert events == expected_events

    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    mojo_auto, mojo_expected = _configured(), _configured()
    actual, events = mojo_auto.simulate(100, 5.0)
    expected, expected_events = mojo_expected.simulate(100, 5.0, backend="mojo")
    np.testing.assert_array_equal(actual, expected)
    assert events == expected_events

    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    rust_auto, rust_expected = DPINeuron(), DPINeuron()
    actual, events = rust_auto.simulate(100, 5.0)
    expected, expected_events = rust_expected.simulate(100, 5.0, backend="rust")
    np.testing.assert_array_equal(actual, expected)
    assert events == expected_events

    monkeypatch.setattr(backends, "_HAS_RUST", False)
    python_auto, python_expected = DPINeuron(), DPINeuron()
    actual, events = python_auto.simulate(100, 5.0)
    expected, expected_events = python_expected.simulate(100, 5.0, backend="python")
    np.testing.assert_array_equal(actual, expected)
    assert events == expected_events


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejects_invalid_run_without_writing_output(backend: str) -> None:
    """Reject invalid work before emitting any caller-visible state."""
    output = np.full(4, -999.0, dtype=np.float64)
    values = (*_factory_values(), 1, math.nan)
    if backend == "go":
        assert backends._go_lib is not None
        result = backends._go_lib.dpi_neuron_simulate_c(
            *values, output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
    else:
        assert backends._mojo_lib is not None
        result = backends._mojo_lib.dpi_neuron_simulate_c(*values, int(output.ctypes.data))
    assert result == -1
    np.testing.assert_array_equal(output, np.full(4, -999.0, dtype=np.float64))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_native_overflow_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate native pre-reset overflow into mutation-free public failure."""
    neuron = DPINeuron(tau=sys.float_info.min)
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, sys.float_info.max, backend=backend)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


@pytest.mark.parametrize("backend", _FULL_CONTRACT_BACKENDS)
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable failure instead of silently falling back."""
    monkeypatch.setattr(backends, f"ensure_{backend}_loaded", lambda: False)
    with pytest.raises(RuntimeError, match=backend.title()):
        DPINeuron().simulate(1, 0.0, backend=backend)


def test_requested_rust_backend_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep explicit Rust selection fail-closed when the extension is absent."""
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    monkeypatch.setattr(backends, "_EngineDPICls", None)
    with pytest.raises(RuntimeError, match="Rust DPI"):
        DPINeuron().simulate(1, 0.0, backend="rust")


def test_dispatcher_runners_reject_missing_loaded_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect direct runner calls as well as the public model boundary."""
    monkeypatch.setattr(backends, "_EngineDPICls", None)
    with pytest.raises(RuntimeError, match="Rust DPI engine"):
        backends.simulate_rust(1, 0.0)
    monkeypatch.setattr(backends, "_julia_module", None)
    with pytest.raises(RuntimeError, match="Julia DPI module"):
        _invoke_full_contract(backends.simulate_julia)
    monkeypatch.setattr(backends, "_go_lib", None)
    with pytest.raises(RuntimeError, match="Go DPI library"):
        _invoke_full_contract(backends.simulate_go)
    monkeypatch.setattr(backends, "_mojo_lib", None)
    with pytest.raises(RuntimeError, match="Mojo DPI library"):
        _invoke_full_contract(backends.simulate_mojo)


def test_direct_c_runner_rejection_names_backend() -> None:
    """Use distinct actionable errors for Go and Mojo ABI rejection."""

    class RejectingLibrary:
        def dpi_neuron_simulate_c(self, *_args: object) -> int:
            return -1

    values = _factory_values()
    with pytest.raises(FloatingPointError, match="Go DPI"):
        backends._simulate_c(RejectingLibrary(), values, 1, 0.0, mojo=False)
    with pytest.raises(FloatingPointError, match="Mojo DPI"):
        backends._simulate_c(RejectingLibrary(), values, 1, 0.0, mojo=True)
