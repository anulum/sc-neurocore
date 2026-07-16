# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable AdEx polyglot backend parity

"""End-to-end parity and rejection contracts for every AdEx acceleration lane."""

from __future__ import annotations

import ctypes
import importlib
import math
import os
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pytest

import sc_neurocore.neurons.models.adex as adex
from sc_neurocore.neurons.models.adex import AdExNeuron

_TRACE_ATOL = 5.0e-12
_GOLDENS = ((0.0, 0), (200.0, 4), (500.0, 12))
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], AdExNeuron] = AdExNeuron,
) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, (neuron.v, neuron.w)


def test_every_acceleration_backend_is_executable() -> None:
    """A graduation run must expose all four real compiled lanes without skips."""
    assert adex._HAS_RUST
    assert adex._ensure_julia_loaded()
    assert adex._ensure_go_loaded()
    assert adex._ensure_mojo_loaded()


def test_missing_rust_engine_is_detected_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the optional-engine import boundary without leaving module drift."""
    original_namespace = adex.__dict__.copy()
    try:
        real_import = importlib.import_module

        def without_engine(name: str, package: str | None = None) -> object:
            if name == "sc_neurocore_engine":
                raise ImportError("engine intentionally hidden")
            return real_import(name, package)

        with monkeypatch.context() as patch:
            patch.setattr(importlib, "import_module", without_engine)
            reloaded = importlib.reload(adex)
            assert reloaded._HAS_RUST is False
            assert reloaded._EngineAdExCls is None
        importlib.reload(adex)
        assert adex._HAS_RUST is True
    finally:
        # Reload mutates the shared module in place and rebinds AdExNeuron.
        # Restore its original namespace so later tests retain their imported
        # class identity instead of observing a stale pre-reload class.
        adex.__dict__.clear()
        adex.__dict__.update(original_namespace)

    assert adex.AdExNeuron is AdExNeuron


def test_invalid_integrator_and_runtime_voltage_fail_closed() -> None:
    """Cover constructor and dynamic-voltage validation boundaries."""
    invalid = cast(Literal["baseline_euler", "rk4", "rosenbrock"], "invalid")
    with pytest.raises(ValueError, match="Unsupported integrator"):
        AdExNeuron(integrator=invalid)

    neuron = AdExNeuron()
    neuron.v = math.nan
    with pytest.raises(ValueError, match="runtime voltage"):
        neuron.simulate(1, 0.0, backend="python")


@pytest.mark.parametrize("current,expected_spikes", _GOLDENS)
@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_compiled_backends_match_python_golden(
    backend: str,
    current: float,
    expected_spikes: int,
) -> None:
    """Preserve the complete trace, final state and stable event observable."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)

    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    np.testing.assert_allclose(trace, reference_trace, atol=_TRACE_ATOL, rtol=0.0)
    np.testing.assert_allclose(state, reference_state, atol=_TRACE_ATOL, rtol=0.0)


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_full_parameter_contract_matches_python(backend: str) -> None:
    """Carry every maintained numeric field across non-default native ABIs."""

    def configured() -> AdExNeuron:
        return AdExNeuron(
            v=-60.0,
            w=3.0,
            v_rest=-64.0,
            v_reset=-69.0,
            v_threshold=-49.0,
            v_rh=-54.0,
            delta_t=2.5,
            tau=18.0,
            tau_w=120.0,
            a=0.7,
            b=8.0,
            c_m=180.0,
            dt=0.2,
        )

    reference_trace, reference_spikes, reference_state = _run(
        "python", current=410.0, n_steps=250, factory=configured
    )
    trace, spikes, state = _run(backend, current=410.0, n_steps=250, factory=configured)
    assert spikes == reference_spikes == 5
    np.testing.assert_allclose(trace, reference_trace, atol=_TRACE_ATOL, rtol=0.0)
    np.testing.assert_allclose(state, reference_state, atol=_TRACE_ATOL, rtol=0.0)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_empty_run_preserves_state(backend: str) -> None:
    """Return an empty trace without discarding either state variable."""
    neuron = AdExNeuron() if backend == "rust" else AdExNeuron(v=-60.0, w=3.0)
    before = (neuron.v, neuron.w)
    trace, spikes = neuron.simulate(0, 250.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert (neuron.v, neuron.w) == before


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_compiled_backends_reject_non_baseline_integrators(backend: str) -> None:
    """Never silently run baseline Euler for an RK4-configured instance."""
    neuron = AdExNeuron(integrator="rk4")
    before = (neuron.v, neuron.w)
    with pytest.raises(RuntimeError, match="baseline_euler"):
        neuron.simulate(1, 0.0, backend=backend)
    assert (neuron.v, neuron.w) == before


def test_rust_rejects_non_default_contract() -> None:
    """Keep the engine class's factory-only parameter boundary explicit."""
    neuron = AdExNeuron(v=-60.0)
    before = (neuron.v, neuron.w)
    with pytest.raises(RuntimeError, match="factory-default"):
        neuron.simulate(1, 0.0, backend="rust")
    assert (neuron.v, neuron.w) == before


def test_auto_uses_python_for_alternative_integrators() -> None:
    """Keep optional RK4 and Rosenbrock semantics independent of baseline kernels."""
    auto = AdExNeuron(integrator="rk4")
    python = AdExNeuron(integrator="rk4")
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    python_trace, python_spikes = python.simulate(100, 250.0, backend="python")
    np.testing.assert_array_equal(auto_trace, python_trace)
    assert (auto_spikes, auto.v, auto.w) == (python_spikes, python.v, python.w)


def test_auto_prefers_measured_fastest_backend() -> None:
    """Route baseline Euler through Mojo before slower compiled lanes."""
    auto = AdExNeuron(v=-60.0, w=3.0)
    expected = AdExNeuron(v=-60.0, w=3.0)
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="mojo")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)


def test_auto_falls_through_to_go_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue through measured order when Mojo and Julia are unavailable."""
    monkeypatch.setattr(adex, "_ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_go_loaded", lambda: True)
    auto = AdExNeuron(v=-60.0, w=3.0)
    expected = AdExNeuron(v=-60.0, w=3.0)
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="go")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)


def test_auto_falls_through_to_julia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Julia when the measured Mojo lane is unavailable."""
    monkeypatch.setattr(adex, "_ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_julia_loaded", lambda: True)
    auto = AdExNeuron(v=-60.0, w=3.0)
    expected = AdExNeuron(v=-60.0, w=3.0)
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="julia")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)


def test_auto_falls_through_to_factory_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Rust after every full-parameter compiled lane is unavailable."""
    monkeypatch.setattr(adex, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_go_loaded", lambda: False)
    auto = AdExNeuron()
    expected = AdExNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="rust")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)


def test_auto_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the Python floor when no compatible compiled lane is available."""
    monkeypatch.setattr(adex, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_go_loaded", lambda: False)
    monkeypatch.setattr(adex, "_HAS_RUST", False)
    auto = AdExNeuron()
    expected = AdExNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="python")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)


def _c_arguments(neuron: AdExNeuron) -> tuple[float, ...]:
    return (
        neuron.v,
        neuron.w,
        neuron.v_rest,
        neuron.v_reset,
        neuron.v_threshold,
        neuron.v_rh,
        neuron.delta_t,
        neuron.tau,
        neuron.tau_w,
        neuron.a,
        neuron.b,
        neuron.c_m,
        neuron.dt,
    )


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejects_non_finite_input_without_writing_output(backend: str) -> None:
    """Prove invalid input is rejected inside each C boundary, not only in Python."""
    neuron = AdExNeuron()
    output = np.full(3, -999.0, dtype=np.float64)
    if backend == "go":
        assert adex._go_lib is not None
        result = adex._go_lib.adex_simulate_c(
            *_c_arguments(neuron),
            1,
            math.nan,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        assert adex._mojo_lib is not None
        result = adex._mojo_lib.adex_simulate_c(
            *_c_arguments(neuron), 1, math.nan, int(output.ctypes.data)
        )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(3, -999.0, dtype=np.float64))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate a native non-finite candidate into a mutation-free failure."""
    neuron = AdExNeuron(dt=1.0e308)
    before = (neuron.v, neuron.w)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, 1.0e308, backend=backend)
    assert (neuron.v, neuron.w) == before


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable failure instead of silently falling back to Python."""
    monkeypatch.setattr(adex, f"_ensure_{backend}_loaded", lambda: False)
    with pytest.raises(RuntimeError, match=backend.title()):
        AdExNeuron().simulate(1, 0.0, backend=backend)


def test_requested_rust_backend_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep explicit Rust requests fail-closed when the engine wheel is absent."""
    monkeypatch.setattr(adex, "_HAS_RUST", False)
    monkeypatch.setattr(adex, "_EngineAdExCls", None)
    with pytest.raises(RuntimeError, match="Rust AdEx backend"):
        AdExNeuron().simulate(1, 0.0, backend="rust")


@pytest.mark.parametrize("backend", ("go", "mojo"))
@pytest.mark.parametrize("failure", ("missing", "load", "symbol"))
def test_c_backend_loader_rejects_invalid_library_boundaries(
    backend: str,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep absent, unloadable or symbol-incomplete libraries unavailable."""
    monkeypatch.setattr(adex, f"_{backend}_lib", None)
    monkeypatch.setattr(adex, f"_HAS_{backend.upper()}", False)
    monkeypatch.setattr(os.path, "isfile", lambda _path: failure != "missing")
    if failure == "load":

        def reject_load(_path: str) -> object:
            raise OSError("invalid shared library")

        monkeypatch.setattr(ctypes, "CDLL", reject_load)
    elif failure == "symbol":
        monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())

    assert getattr(adex, f"_ensure_{backend}_loaded")() is False
    assert getattr(adex, f"_{backend}_lib") is None
    assert getattr(adex, f"_HAS_{backend.upper()}") is False


@pytest.mark.parametrize("failure", ("missing", "load", "module"))
def test_julia_loader_rejects_invalid_runtime_boundaries(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep missing runtimes, source files and broken modules unavailable."""
    monkeypatch.setattr(adex, "_julia_module", None)
    monkeypatch.setattr(adex, "_HAS_JULIA", False)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda _name: None if failure == "missing" else 1
    )
    monkeypatch.setattr(os.path, "isfile", lambda _path: failure != "load")
    if failure == "module":

        def reject_import(_name: str) -> object:
            raise RuntimeError("broken Julia runtime")

        monkeypatch.setattr(importlib, "import_module", reject_import)

    assert adex._ensure_julia_loaded() is False
    assert adex._julia_module is None
    assert adex._HAS_JULIA is False
