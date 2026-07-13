# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable ExpIF polyglot backend parity

"""End-to-end parity and rejection contracts for every ExpIF native lane."""

from __future__ import annotations

import ctypes
import importlib
import math
import os
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import sc_neurocore.neurons.models.expif as expif
from sc_neurocore.neurons.models.expif import ExpIFNeuron

_TRACE_ATOL = 5.0e-8
_GOLDENS = ((0.0, 0), (5.0, 0), (20.0, 2), (50.0, 5))
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], ExpIFNeuron] = ExpIFNeuron,
) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
    """Run one backend and return its trace, event count, and final state."""
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, (neuron.v, neuron.refractory_remaining)


def _configured() -> ExpIFNeuron:
    """Return a non-default state that exercises the complete native ABI."""
    return ExpIFNeuron(
        v=-62.0,
        v_rest=-64.0,
        v_reset=-69.0,
        v_threshold=25.0,
        v_rh=-58.0,
        delta_t=3.0,
        tau=12.0,
        dt=0.03,
        refractory_period=0.09,
        refractory_remaining=0.06,
    )


def test_every_acceleration_backend_is_executable() -> None:
    """A fidelity-closure run exposes all four real compiled lanes without skips."""
    assert expif._HAS_RUST
    assert expif._ensure_julia_loaded()
    assert expif._ensure_go_loaded()
    assert expif._ensure_mojo_loaded()


def test_missing_rust_engine_is_detected_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the optional-engine import boundary without leaving module drift."""
    real_import = importlib.import_module

    def without_engine(name: str, package: str | None = None) -> object:
        if name == "sc_neurocore_engine":
            raise ImportError("engine intentionally hidden")
        return real_import(name, package)

    with monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", without_engine)
        reloaded = importlib.reload(expif)
        assert reloaded._HAS_RUST is False
        assert reloaded._EngineExpIFCls is None
    importlib.reload(expif)
    assert expif._HAS_RUST is True


@pytest.mark.parametrize(("current", "expected_spikes"), _GOLDENS)
@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_compiled_backends_match_python_golden(
    backend: str,
    current: float,
    expected_spikes: int,
) -> None:
    """Preserve the complete trace, final state, and source-bound events."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)

    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    np.testing.assert_allclose(trace, reference_trace, atol=_TRACE_ATOL, rtol=0.0)
    np.testing.assert_allclose(state, reference_state, atol=_TRACE_ATOL, rtol=0.0)


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_full_parameter_and_refractory_contract_matches_python(backend: str) -> None:
    """Carry every maintained numeric field across full-parameter native ABIs."""
    reference_trace, reference_spikes, reference_state = _run(
        "python", current=50.0, n_steps=500, factory=_configured
    )
    trace, spikes, state = _run(backend, current=50.0, n_steps=500, factory=_configured)
    assert spikes == reference_spikes == 2
    np.testing.assert_allclose(trace, reference_trace, atol=_TRACE_ATOL, rtol=0.0)
    np.testing.assert_allclose(state, reference_state, atol=_TRACE_ATOL, rtol=0.0)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_empty_run_preserves_state(backend: str) -> None:
    """Return an empty trace without discarding voltage or refractory state."""
    neuron = ExpIFNeuron() if backend == "rust" else _configured()
    before = (neuron.v, neuron.refractory_remaining)
    trace, spikes = neuron.simulate(0, 20.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert (neuron.v, neuron.refractory_remaining) == before


def test_rust_rejects_non_default_contract() -> None:
    """Keep the Rust engine class's factory-only parameter boundary explicit."""
    neuron = ExpIFNeuron(v=-60.0)
    before = (neuron.v, neuron.refractory_remaining)
    with pytest.raises(RuntimeError, match="factory-default"):
        neuron.simulate(1, 0.0, backend="rust")
    assert (neuron.v, neuron.refractory_remaining) == before


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


def test_auto_prefers_first_full_parameter_backend() -> None:
    """Route a non-default instance through measured-first Julia."""
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 50.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 50.0, backend="julia")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )


def test_auto_falls_through_to_go(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Go when the measured-first Julia lane is unavailable."""
    monkeypatch.setattr(expif, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_go_loaded", lambda: True)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 50.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 50.0, backend="go")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )


def test_auto_falls_through_to_mojo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue to Mojo when Julia and Go are unavailable."""
    monkeypatch.setattr(expif, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_go_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_mojo_loaded", lambda: True)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 50.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 50.0, backend="mojo")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )


def test_auto_falls_through_to_factory_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Rust when every full-parameter compiled lane is unavailable."""
    monkeypatch.setattr(expif, "_ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_go_loaded", lambda: False)
    auto = ExpIFNeuron()
    expected = ExpIFNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 20.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 20.0, backend="rust")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )


def test_auto_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the Python floor when no compatible compiled lane is available."""
    monkeypatch.setattr(expif, "_ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_go_loaded", lambda: False)
    monkeypatch.setattr(expif, "_HAS_RUST", False)
    auto = ExpIFNeuron()
    expected = ExpIFNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 20.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 20.0, backend="python")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )


def _c_arguments(neuron: ExpIFNeuron) -> tuple[float, ...]:
    """Return numeric fields in the C-ABI declaration order."""
    return (
        neuron.v,
        neuron.v_rest,
        neuron.v_reset,
        neuron.v_threshold,
        neuron.v_rh,
        neuron.delta_t,
        neuron.tau,
        neuron.dt,
        neuron.refractory_period,
        neuron.refractory_remaining,
    )


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejects_non_finite_input_without_writing_output(backend: str) -> None:
    """Prove invalid input is rejected inside each C boundary before emission."""
    neuron = ExpIFNeuron()
    output = np.full(3, -999.0, dtype=np.float64)
    if backend == "go":
        assert expif._go_lib is not None
        result = expif._go_lib.expif_simulate_c(
            *_c_arguments(neuron),
            1,
            math.nan,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        assert expif._mojo_lib is not None
        result = expif._mojo_lib.expif_simulate_c(
            *_c_arguments(neuron), 1, math.nan, int(output.ctypes.data)
        )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(3, -999.0, dtype=np.float64))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate a native non-finite candidate into a mutation-free failure."""
    neuron = ExpIFNeuron(dt=1.0e308)
    before = (neuron.v, neuron.refractory_remaining)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, 1.0e308, backend=backend)
    assert (neuron.v, neuron.refractory_remaining) == before


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable failure instead of silently falling back to Python."""
    monkeypatch.setattr(expif, f"_ensure_{backend}_loaded", lambda: False)
    with pytest.raises(RuntimeError, match=backend.title()):
        ExpIFNeuron().simulate(1, 0.0, backend=backend)


def test_requested_rust_backend_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep explicit Rust requests fail-closed when the engine wheel is absent."""
    monkeypatch.setattr(expif, "_HAS_RUST", False)
    monkeypatch.setattr(expif, "_EngineExpIFCls", None)
    with pytest.raises(RuntimeError, match="Rust ExpIF backend"):
        ExpIFNeuron().simulate(1, 0.0, backend="rust")


@pytest.mark.parametrize("backend", ("go", "mojo"))
@pytest.mark.parametrize("failure", ("missing", "load", "symbol"))
def test_c_backend_loader_rejects_invalid_library_boundaries(
    backend: str,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep absent, unloadable, or symbol-incomplete libraries unavailable."""
    monkeypatch.setattr(expif, f"_{backend}_lib", None)
    monkeypatch.setattr(expif, f"_HAS_{backend.upper()}", False)
    monkeypatch.setattr(os.path, "isfile", lambda _path: failure != "missing")
    if failure == "load":

        def reject_load(_path: str) -> object:
            raise OSError("invalid shared library")

        monkeypatch.setattr(ctypes, "CDLL", reject_load)
    elif failure == "symbol":
        monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())

    assert getattr(expif, f"_ensure_{backend}_loaded")() is False
    assert getattr(expif, f"_{backend}_lib") is None
    assert getattr(expif, f"_HAS_{backend.upper()}") is False


@pytest.mark.parametrize("failure", ("missing", "source", "module"))
def test_julia_loader_rejects_invalid_runtime_boundaries(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep missing runtimes, source files, and broken modules unavailable."""
    monkeypatch.setattr(expif, "_julia_module", None)
    monkeypatch.setattr(expif, "_HAS_JULIA", False)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda _name: None if failure == "missing" else 1
    )
    monkeypatch.setattr(os.path, "isfile", lambda _path: failure != "source")
    if failure == "module":

        def reject_import(_name: str) -> object:
            raise RuntimeError("broken Julia runtime")

        monkeypatch.setattr(importlib, "import_module", reject_import)

    assert expif._ensure_julia_loaded() is False
    assert expif._julia_module is None
    assert expif._HAS_JULIA is False
