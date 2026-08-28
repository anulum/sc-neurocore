# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained scaled-reset adaptive IF five-runtime parity

"""Complete-state parity for the count-neutral scaled-reset recurrence."""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models import sc_scaled_reset_adaptive_if
from sc_neurocore.neurons.models.sc_scaled_reset_adaptive_if import (
    SCScaledResetAdaptiveIFNeuron,
)


def _run(
    backend: str,
    *,
    steps: int = 4000,
    current: float = 2.0,
    parameters: Mapping[str, float] | None = None,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], int, tuple[float, ...]]:
    neuron = SCScaledResetAdaptiveIFNeuron(**({} if parameters is None else parameters))
    trace, events = neuron.simulate(steps, current, backend=backend)
    return trace, events, (neuron.v, neuron.theta, neuron.i1, neuron.i2)


def _rust() -> bool:
    return sc_scaled_reset_adaptive_if._HAS_RUST


def _julia() -> bool:
    return sc_scaled_reset_adaptive_if._ensure_julia_loaded()


def _go() -> bool:
    return sc_scaled_reset_adaptive_if._ensure_go_loaded()


def _mojo() -> bool:
    return sc_scaled_reset_adaptive_if._ensure_mojo_loaded()


_EXACT: tuple[tuple[str, Callable[[], bool]], ...] = (
    ("rust", _rust),
    ("julia", _julia),
    ("go", _go),
)
_CURRENTS = (0.0, 1.5, 2.0, 3.0)
_REGIMES: tuple[dict[str, float], ...] = (
    {},
    {"a": 0.5, "b": 0.05, "r1": 0.5, "r2": 0.3},
    {"a": 1.0, "b": 0.1, "theta_inf": 0.8, "r1": 1.0, "r2": 0.2},
    {"tau_v": 5.0, "tau_theta": 50.0, "a": 0.3, "b": 0.2},
)


@pytest.mark.parametrize("backend,available", _EXACT, ids=[name for name, _ in _EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_exact_backends_match_complete_state(
    backend: str, available: Callable[[], bool], current: float
) -> None:
    if not available():
        pytest.skip(f"{backend} retained backend unavailable")
    reference = _run("python", current=current)
    observed = _run(backend, current=current)

    np.testing.assert_array_equal(observed[0], reference[0])
    assert observed[1:] == reference[1:]


@pytest.mark.parametrize("backend,available", _EXACT, ids=[name for name, _ in _EXACT])
@pytest.mark.parametrize(
    "parameters", _REGIMES, ids=("default", "kicks", "threshold", "timescales")
)
def test_exact_backends_match_retained_regimes(
    backend: str,
    available: Callable[[], bool],
    parameters: dict[str, float],
) -> None:
    if not available():
        pytest.skip(f"{backend} retained backend unavailable")
    reference = _run("python", parameters=parameters)
    observed = _run(backend, parameters=parameters)

    np.testing.assert_array_equal(observed[0], reference[0])
    assert observed[1:] == reference[1:]


@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_is_event_exact_and_ulp_bounded(current: float) -> None:
    if not _mojo():
        pytest.skip("Mojo retained backend unavailable")
    reference = _run("python", steps=20_000, current=current)
    observed = _run("mojo", steps=20_000, current=current)

    np.testing.assert_allclose(observed[0], reference[0], rtol=0.0, atol=1e-9)
    assert observed[1] == reference[1]
    np.testing.assert_allclose(observed[2], reference[2], rtol=0.0, atol=1e-9)


def test_auto_dispatch_matches_explicit_python() -> None:
    reference = _run("python")
    observed = _run("auto")

    np.testing.assert_allclose(observed[0], reference[0], rtol=0.0, atol=1e-9)
    assert observed[1:] == reference[1:]


def test_public_dispatch_rejects_invalid_backend_current_and_steps() -> None:
    neuron = SCScaledResetAdaptiveIFNeuron()
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, backend="cuda")
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, math.nan, backend="python")
    invalid_steps: tuple[Any, ...] = (-1, True, 1.5, 1 << 31)
    for steps in invalid_steps:
        with pytest.raises(ValueError, match="n_steps"):
            neuron.simulate(steps, 2.0)


@pytest.mark.parametrize(
    ("trace", "events", "state", "message"),
    (
        (np.array([], dtype=np.float64), 0, (0.0, 1.0, 0.0, 0.0), "trace length"),
        (np.array([math.inf]), 0, (0.0, 1.0, 0.0, 0.0), "non-finite"),
        (np.array([0.0]), -1, (0.0, 1.0, 0.0, 0.0), "event count"),
    ),
)
def test_backend_result_validation_is_failure_atomic(
    monkeypatch: pytest.MonkeyPatch,
    trace: np.ndarray[Any, np.dtype[np.float64]],
    events: int,
    state: tuple[float, float, float, float],
    message: str,
) -> None:
    neuron = SCScaledResetAdaptiveIFNeuron(v=0.2)
    before = (neuron.v, neuron.theta, neuron.i1, neuron.i2)
    monkeypatch.setattr(neuron, "_simulate_python", lambda *_: (trace, events, state))

    with pytest.raises(FloatingPointError, match=message):
        neuron.simulate(1, 2.0, backend="python")

    assert (neuron.v, neuron.theta, neuron.i1, neuron.i2) == before


def test_public_rust_batch_dispatch_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = SCScaledResetAdaptiveIFNeuron()
    expected_trace, expected_events = reference.simulate(3, 2.0, backend="python")
    expected_state = (reference.v, reference.theta, reference.i1, reference.i2)
    monkeypatch.setattr(sc_scaled_reset_adaptive_if, "_HAS_RUST", True)
    monkeypatch.setattr(
        sc_scaled_reset_adaptive_if,
        "_rust_simulate",
        lambda *_: (expected_trace.tolist(), expected_events, *expected_state),
    )

    neuron = SCScaledResetAdaptiveIFNeuron()
    trace, events = neuron.simulate(3, 2.0, backend="rust")

    np.testing.assert_array_equal(trace, expected_trace)
    assert events == expected_events
    assert (neuron.v, neuron.theta, neuron.i1, neuron.i2) == expected_state


@pytest.mark.parametrize(
    ("backend", "availability_name"),
    (
        ("julia", "_ensure_julia_loaded"),
        ("go", "_ensure_go_loaded"),
        ("mojo", "_ensure_mojo_loaded"),
    ),
)
def test_explicit_optional_backend_reports_unavailability(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    availability_name: str,
) -> None:
    monkeypatch.setattr(sc_scaled_reset_adaptive_if, availability_name, lambda: False)

    with pytest.raises(RuntimeError, match=backend.capitalize()):
        SCScaledResetAdaptiveIFNeuron().simulate(1, 2.0, backend=backend)


def test_explicit_rust_backend_reports_unavailability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc_scaled_reset_adaptive_if, "_HAS_RUST", False)

    with pytest.raises(RuntimeError, match="Rust"):
        SCScaledResetAdaptiveIFNeuron().simulate(1, 2.0, backend="rust")


@pytest.mark.parametrize(
    ("backend", "cache_name"),
    (("julia", "_julia_module"), ("go", "_go_lib"), ("mojo", "_mojo_lib")),
)
def test_public_dispatch_reports_missing_optional_library(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    cache_name: str,
) -> None:
    monkeypatch.setattr(sc_scaled_reset_adaptive_if, cache_name, None)
    monkeypatch.setattr(os.path, "isfile", lambda _: False)
    if backend == "julia":
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())

    with pytest.raises(RuntimeError, match=backend.capitalize()):
        SCScaledResetAdaptiveIFNeuron().simulate(1, 2.0, backend=backend)


@pytest.mark.parametrize(("backend", "cache_name"), (("go", "_go_lib"), ("mojo", "_mojo_lib")))
@pytest.mark.parametrize("failure", ("load", "symbol"))
def test_public_dispatch_reports_invalid_native_library(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    cache_name: str,
    failure: str,
) -> None:
    monkeypatch.setattr(sc_scaled_reset_adaptive_if, cache_name, None)
    monkeypatch.setattr(os.path, "isfile", lambda _: True)

    def load_library(_: str) -> object:
        if failure == "load":
            raise OSError("invalid shared object")
        return object()

    monkeypatch.setattr(ctypes, "CDLL", load_library)

    with pytest.raises(RuntimeError, match=backend.capitalize()):
        SCScaledResetAdaptiveIFNeuron().simulate(1, 2.0, backend=backend)


def test_public_dispatch_reports_missing_julia_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sc_scaled_reset_adaptive_if, "_julia_module", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)

    with pytest.raises(RuntimeError, match="Julia"):
        SCScaledResetAdaptiveIFNeuron().simulate(1, 2.0, backend="julia")


@pytest.mark.parametrize("backend,available", (("go", _go), ("mojo", _mojo)))
def test_native_zero_step_preserves_state(backend: str, available: Callable[[], bool]) -> None:
    if not available():
        pytest.skip(f"{backend} retained backend unavailable")
    neuron = SCScaledResetAdaptiveIFNeuron(v=0.2, theta=1.2, i1=0.3, i2=-0.1)
    before = (neuron.v, neuron.theta, neuron.i1, neuron.i2)

    trace, events = neuron.simulate(0, 2.0, backend=backend)

    assert trace.size == 0
    assert events == 0
    assert (neuron.v, neuron.theta, neuron.i1, neuron.i2) == before
