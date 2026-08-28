# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev 2005 first-return polyglot parity

"""Cross-language parity for the calibrated slow-calcium recurrence."""

from __future__ import annotations

import ctypes
import importlib.util
import os
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models import medvedev_map
from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron

Availability = Callable[[], bool]
_BACKENDS: tuple[tuple[str, Availability], ...] = (
    ("rust", lambda: medvedev_map._HAS_RUST),
    ("julia", medvedev_map._ensure_julia_loaded),
    ("go", medvedev_map._ensure_go_loaded),
    ("mojo", medvedev_map._ensure_mojo_loaded),
)
_COMPILED_ABS_TOL = 5.0e-13


def _run(
    backend: str,
    *,
    u0: float = 0.2514078836724436,
    n_steps: int = 1000,
    current: float = 2.0,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run one backend and return its trace, event count and final state."""
    neuron = MedvedevMapNeuron(u=u0)
    trace, events = neuron.simulate(n_steps, current, backend=backend)
    return trace, events, neuron.u


@pytest.mark.parametrize("backend,available", _BACKENDS, ids=[name for name, _ in _BACKENDS])
@pytest.mark.parametrize("current", (0.0, 2.0, 16.0))
def test_trace_events_and_final_state_are_ulp_bounded(
    backend: str,
    available: Availability,
    current: float,
) -> None:
    """Every compiled lane tracks the source recurrence and exact events."""
    if not available():
        pytest.skip(f"{backend} Medvedev backend unavailable")
    expected_trace, expected_events, expected_final = _run("python", current=current)
    trace, events, final_state = _run(backend, current=current)
    np.testing.assert_allclose(trace, expected_trace, rtol=0.0, atol=_COMPILED_ABS_TOL)
    assert events == expected_events
    assert final_state == pytest.approx(expected_final, rel=0.0, abs=_COMPILED_ABS_TOL)


@pytest.mark.parametrize("backend,available", _BACKENDS, ids=[name for name, _ in _BACKENDS])
def test_empty_single_and_right_branch_requests(
    backend: str,
    available: Availability,
) -> None:
    """ABI boundaries and the current-free Eq. 4.15 branch retain parity."""
    if not available():
        pytest.skip(f"{backend} Medvedev backend unavailable")
    for n_steps, u0, current in ((0, 0.2, 2.0), (1, 0.2, 2.0), (1, 0.3, 1000.0)):
        expected = _run("python", u0=u0, n_steps=n_steps, current=current)
        observed = _run(backend, u0=u0, n_steps=n_steps, current=current)
        np.testing.assert_allclose(observed[0], expected[0], rtol=0.0, atol=_COMPILED_ABS_TOL)
        assert observed[1] == expected[1]
        assert observed[2] == pytest.approx(expected[2], rel=0.0, abs=_COMPILED_ABS_TOL)


def test_auto_dispatch_matches_source_contract() -> None:
    """Measured-order dispatch cannot change the numerical contract."""
    expected_trace, expected_events, expected_final = _run("python")
    trace, events, final_state = _run("auto")
    np.testing.assert_allclose(trace, expected_trace, rtol=0.0, atol=_COMPILED_ABS_TOL)
    assert events == expected_events
    assert final_state == pytest.approx(expected_final, rel=0.0, abs=_COMPILED_ABS_TOL)


def test_explicit_unavailable_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit compiled-lane request never silently falls back."""
    monkeypatch.setattr(medvedev_map, "_HAS_RUST", False)
    monkeypatch.setattr(medvedev_map, "_rust_simulate", None)
    with pytest.raises(RuntimeError, match="Rust Medvedev backend unavailable"):
        MedvedevMapNeuron().simulate(1, 2.0, backend="rust")


@pytest.mark.parametrize(
    "backend,loader,message",
    (
        ("go", "_ensure_go_loaded", "Go Medvedev backend unavailable"),
        ("mojo", "_ensure_mojo_loaded", "Mojo Medvedev backend unavailable"),
        ("julia", "_ensure_julia_loaded", "Julia Medvedev backend unavailable"),
    ),
)
def test_each_explicit_unavailable_backend_identifies_its_build_boundary(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    loader: str,
    message: str,
) -> None:
    """Each maintained lane fails closed with its own actionable diagnostic."""
    monkeypatch.setattr(medvedev_map, loader, lambda: False)
    with pytest.raises(RuntimeError, match=message):
        MedvedevMapNeuron().simulate(1, 2.0, backend=backend)


def test_optional_runtime_loader_rejections_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing runtimes, binaries, and ABI symbols cannot become false availability."""
    monkeypatch.setattr(medvedev_map, "_julia_module", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert not medvedev_map._ensure_julia_loaded()

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(os.path, "isfile", lambda _path: False)
    assert not medvedev_map._ensure_julia_loaded()
    assert medvedev_map._load_c_backend("missing.so", mojo=False) is None

    monkeypatch.setattr(os.path, "isfile", lambda _path: True)

    def raise_os_error(_path: str) -> None:
        raise OSError("invalid shared object")

    monkeypatch.setattr(ctypes, "CDLL", raise_os_error)
    assert medvedev_map._load_c_backend("invalid.so", mojo=False) is None

    monkeypatch.setattr(ctypes, "CDLL", lambda _path: SimpleNamespace())
    assert medvedev_map._load_c_backend("wrong-abi.so", mojo=True) is None


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_native_backend_rejection_preserves_python_state(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """A negative native status is surfaced before the Python state commits."""
    rejecting = SimpleNamespace(medvedev_map_simulate_c=lambda *_args: -1)
    library_name = "_go_lib" if backend == "go" else "_mojo_lib"
    monkeypatch.setattr(medvedev_map, library_name, rejecting)

    neuron = MedvedevMapNeuron()
    before = neuron.u
    with pytest.raises(FloatingPointError, match=f"{backend.title()} Medvedev backend rejected"):
        neuron.simulate(1, 2.0, backend=backend)
    assert neuron.u == before


@pytest.mark.parametrize("backend", ("python", "rust", "julia", "go", "mojo"))
def test_invalid_current_fails_before_backend_mutation(backend: str) -> None:
    """Front-door validation rejects NaN before any backend writes state."""
    neuron = MedvedevMapNeuron()
    before = neuron.u
    with pytest.raises(ValueError, match="current must be finite"):
        neuron.simulate(10, float("nan"), backend=backend)
    assert neuron.u == before
