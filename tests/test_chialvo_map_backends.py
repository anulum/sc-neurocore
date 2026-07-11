# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cross-language parity for the Chialvo map

"""Parity and dispatch contracts for ``ChialvoMapNeuron.simulate``.

The recurrence contains ``exp`` and is chaotic in part of its parameter space.
Independent libm implementations therefore need not reproduce a long trace
bit-for-bit. The contract is source-equation one-step agreement, identical
maintained event counts at the enrolled operating set, and an explicitly
measured trajectory envelope. Mojo's optimised arithmetic has the widest bound.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models import chialvo_map
from sc_neurocore.neurons.models.chialvo_map import ChialvoMapNeuron

Availability = Callable[[], bool]
RunResult = tuple[npt.NDArray[np.float64], int, float, float]


def _rust_available() -> bool:
    return bool(chialvo_map._HAS_RUST)


def _julia_available() -> bool:
    return bool(chialvo_map._ensure_julia_loaded())


def _go_available() -> bool:
    return bool(chialvo_map._ensure_go_loaded())


def _mojo_available() -> bool:
    return bool(chialvo_map._ensure_mojo_loaded())


_TRACE_TOLERANCE = {"rust": 5e-14, "julia": 5e-14, "go": 5e-14, "mojo": 2e-9}
_STEP_TOLERANCE = {"rust": 5e-15, "julia": 5e-15, "go": 5e-15, "mojo": 5e-11}
_OPERATING_CURRENTS = (-0.05, 0.0, 0.01, 0.05, 0.1, 1.0)


def _require(backend: str, available: Availability) -> None:
    if not available():
        pytest.skip(f"{backend} Chialvo backend is not built in this environment")


def _run(
    backend: str,
    *,
    n_steps: int = 1000,
    current: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
) -> RunResult:
    neuron = ChialvoMapNeuron(x=x, y=y)
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, neuron.x, neuron.y


def _assert_source_equation_one_step_envelope(backend: str) -> None:
    rng = np.random.default_rng(20260711)
    tolerance = _STEP_TOLERANCE[backend]
    for _sample in range(1000):
        x = float(rng.uniform(-1.0, 1.5))
        y = float(rng.uniform(-1.0, 2.5))
        current = float(rng.uniform(-0.1, 0.2))
        reference = _run("python", n_steps=1, current=current, x=x, y=y)
        observed = _run(backend, n_steps=1, current=current, x=x, y=y)
        np.testing.assert_allclose(observed[0], reference[0], atol=tolerance, rtol=0.0)
        assert observed[1] == reference[1]
        assert observed[2] == pytest.approx(reference[2], abs=tolerance)
        assert observed[3] == pytest.approx(reference[3], abs=tolerance)


def _assert_enrolled_event_counts_and_trace_envelope(backend: str) -> None:
    for current in _OPERATING_CURRENTS:
        reference = _run("python", current=current)
        observed = _run(backend, current=current)
        tolerance = _TRACE_TOLERANCE[backend]
        assert observed[1] == reference[1]
        np.testing.assert_allclose(observed[0], reference[0], atol=tolerance, rtol=0.0)
        assert observed[2] == pytest.approx(reference[2], abs=tolerance)
        assert observed[3] == pytest.approx(reference[3], abs=tolerance)


def _assert_empty_and_single_step_preserve_state_contract(backend: str) -> None:
    for n_steps in (0, 1):
        reference = _run("python", n_steps=n_steps, current=0.01, x=0.2, y=0.7)
        observed = _run(backend, n_steps=n_steps, current=0.01, x=0.2, y=0.7)
        tolerance = _STEP_TOLERANCE[backend]
        np.testing.assert_allclose(observed[0], reference[0], atol=tolerance, rtol=0.0)
        assert observed[1] == reference[1]
        assert observed[2] == pytest.approx(reference[2], abs=tolerance)
        assert observed[3] == pytest.approx(reference[3], abs=tolerance)


def _assert_backend_contract(backend: str, available: Availability) -> None:
    _require(backend, available)
    _assert_source_equation_one_step_envelope(backend)
    _assert_enrolled_event_counts_and_trace_envelope(backend)
    _assert_empty_and_single_step_preserve_state_contract(backend)


def test_rust_backend_contract() -> None:
    """Rust must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("rust", _rust_available)


def test_julia_backend_contract() -> None:
    """Julia must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("julia", _julia_available)


def test_go_backend_contract() -> None:
    """Go must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("go", _go_available)


def test_mojo_backend_contract() -> None:
    """Mojo must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("mojo", _mojo_available)


def test_python_batch_matches_repeated_step() -> None:
    """The reference batch loop must retain the public step semantics."""
    batch = ChialvoMapNeuron()
    trace, spikes = batch.simulate(100, 0.01, backend="python")
    stepper = ChialvoMapNeuron()
    expected: npt.NDArray[np.float64] = np.empty(100, dtype=np.float64)
    expected_spikes = 0
    for index in range(100):
        expected_spikes += stepper.step(0.01)
        expected[index] = stepper.x
    np.testing.assert_array_equal(trace, expected)
    assert spikes == expected_spikes
    assert (batch.x, batch.y) == (stepper.x, stepper.y)


def test_auto_uses_measured_first_available_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto dispatch must honour the data-driven order without hidden fallback."""
    visited: list[str] = []

    def order(_kernel: str, *, static: tuple[str, ...]) -> tuple[str, ...]:
        assert static == chialvo_map._AUTO_BACKENDS
        return ("go", "mojo", "rust", "julia", "python")

    def available(name: str) -> bool:
        visited.append(name)
        return name == "rust"

    def rust_simulate(_neuron: ChialvoMapNeuron, n_steps: int, current: float) -> RunResult:
        assert n_steps == 1
        assert current == 0.0
        return np.array([0.5], dtype=np.float64), 0, 0.5, 0.0

    monkeypatch.setattr(chialvo_map, "select_backend_order", order)
    monkeypatch.setattr(chialvo_map, "_backend_available", available)
    monkeypatch.setattr(ChialvoMapNeuron, "_simulate_rust", rust_simulate)
    trace, spikes = ChialvoMapNeuron().simulate(1, backend="auto")
    np.testing.assert_array_equal(trace, np.array([0.5], dtype=np.float64))
    assert spikes == 0
    assert visited == ["go", "mojo", "rust", "rust"]


def test_auto_python_floor_matches_explicit_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guaranteed auto floor must be the checked Python recurrence."""
    monkeypatch.setattr(chialvo_map, "_auto_backend", lambda: "python")
    explicit = _run("python", n_steps=100, current=0.01)
    automatic = _run("auto", n_steps=100, current=0.01)
    np.testing.assert_array_equal(automatic[0], explicit[0])
    assert automatic[1:] == explicit[1:]


def test_explicit_unavailable_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit lane request must never fall through to Python silently."""
    monkeypatch.setattr(chialvo_map, "_backend_available", lambda _name: False)
    for backend in ("rust", "julia", "go", "mojo"):
        with pytest.raises(RuntimeError, match="unavailable"):
            ChialvoMapNeuron().simulate(1, backend=backend)


def test_import_without_rust_extension_keeps_python_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional Rust extension must not prevent Python use."""
    real_import = importlib.import_module

    def import_without_engine(name: str, package: str | None = None) -> object:
        if name == "sc_neurocore_engine":
            raise ImportError("extension absent")
        return real_import(name, package)

    with monkeypatch.context() as context:
        context.setattr(importlib, "import_module", import_without_engine)
        reloaded = importlib.reload(chialvo_map)
        trace, spikes = reloaded.ChialvoMapNeuron().simulate(2, backend="python")
        assert trace.shape == (2,)
        assert spikes in (0, 1, 2)
        with pytest.raises(RuntimeError, match="unavailable"):
            reloaded.ChialvoMapNeuron().simulate(1, backend="rust")
    importlib.reload(chialvo_map)


def test_julia_loader_reports_missing_runtime_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Julia selection must fail closed when either required surface is absent."""
    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_julia_module", None)
        context.setattr(importlib.util, "find_spec", lambda _name: None)
        with pytest.raises(RuntimeError, match="unavailable"):
            ChialvoMapNeuron().simulate(1, backend="julia")

    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_julia_module", None)
        context.setattr(importlib.util, "find_spec", lambda _name: object())
        context.setattr(os.path, "isfile", lambda _path: False)
        with pytest.raises(RuntimeError, match="unavailable"):
            ChialvoMapNeuron().simulate(1, backend="julia")


def test_c_loader_reports_missing_file_load_failure_and_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Go selection must reject each shared-library discovery failure."""
    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_go_lib", None)
        context.setattr(os.path, "isfile", lambda _path: False)
        with pytest.raises(RuntimeError, match="Go Chialvo backend unavailable"):
            ChialvoMapNeuron().simulate(1, backend="go")

    def reject_library(_path: str) -> ctypes.CDLL:
        raise OSError("invalid shared object")

    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_go_lib", None)
        context.setattr(os.path, "isfile", lambda _path: True)
        context.setattr(ctypes, "CDLL", reject_library)
        with pytest.raises(RuntimeError, match="Go Chialvo backend unavailable"):
            ChialvoMapNeuron().simulate(1, backend="go")

    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_go_lib", None)
        context.setattr(os.path, "isfile", lambda _path: True)
        context.setattr(ctypes, "CDLL", lambda _path: object())
        with pytest.raises(RuntimeError, match="Go Chialvo backend unavailable"):
            ChialvoMapNeuron().simulate(1, backend="go")


def test_auto_empty_order_and_loaded_backend_loss_keep_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto retains its floor, while stale compiled handles remain explicit errors."""
    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "select_backend_order", lambda *_args, **_kwargs: ())
        trace, _spikes = ChialvoMapNeuron().simulate(2, backend="auto")
        assert trace.shape == (2,)

    with monkeypatch.context() as context:
        context.setattr(
            chialvo_map,
            "select_backend_order",
            lambda *_args, **_kwargs: ("unknown", "python"),
        )
        trace, _spikes = ChialvoMapNeuron().simulate(2, backend="auto")
        assert trace.shape == (2,)

    for backend, attribute in (
        ("rust", "_rust_simulate"),
        ("julia", "_julia_module"),
        ("go", "_go_lib"),
        ("mojo", "_mojo_lib"),
    ):
        with monkeypatch.context() as context:
            context.setattr(chialvo_map, "_backend_available", lambda _name: True)
            context.setattr(chialvo_map, attribute, None)
            with pytest.raises(RuntimeError, match="unavailable"):
                ChialvoMapNeuron().simulate(1, backend=backend)


def test_compiled_error_sentinel_becomes_floating_point_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C ABI numerical rejection must not expose an uninitialised trace."""

    class RejectingFunction:
        def __call__(self, *_args: object) -> int:
            return -1

    class RejectingLibrary:
        chialvo_map_simulate_c = RejectingFunction()

    monkeypatch.setattr(chialvo_map, "_backend_available", lambda _name: True)
    monkeypatch.setattr(chialvo_map, "_mojo_lib", RejectingLibrary())
    with pytest.raises(FloatingPointError, match="rejected"):
        ChialvoMapNeuron().simulate(1, backend="mojo")


def test_invalid_backend_rejected() -> None:
    for backend in ("cuda", "", "RUST"):
        with pytest.raises(ValueError, match="backend must be"):
            ChialvoMapNeuron().simulate(1, backend=backend)


def test_invalid_batch_arguments_and_mutable_configuration_rejected() -> None:
    neuron = ChialvoMapNeuron()
    with pytest.raises(ValueError, match="non-negative"):
        neuron.simulate(-1)
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, current=np.inf)
    neuron.k = np.nan
    with pytest.raises(ValueError, match="k"):
        neuron.simulate(1)


def test_reset_preserves_configuration() -> None:
    neuron = ChialvoMapNeuron(a=0.8, b=0.4, c=0.2, k=0.03, x_threshold=0.75)
    neuron.step(0.01)
    neuron.reset()
    assert (neuron.x, neuron.y) == (0.0, 0.0)
    assert (neuron.a, neuron.b, neuron.c, neuron.k, neuron.x_threshold) == (
        0.8,
        0.4,
        0.2,
        0.03,
        0.75,
    )
