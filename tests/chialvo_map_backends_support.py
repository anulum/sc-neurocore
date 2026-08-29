# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_chialvo_map_backends.py

from __future__ import annotations

"""Parity and dispatch contracts for ``ChialvoMapNeuron.simulate``.

The recurrence contains ``exp`` and is chaotic in part of its parameter space.
Independent libm implementations therefore need not reproduce a long trace
bit-for-bit. The contract is source-equation one-step agreement, identical
maintained event counts at the enrolled operating set, and an explicitly
measured trajectory envelope. Mojo's optimised arithmetic has the widest bound."""

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

RunResult = tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    int,
    float,
    float,
]


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
    x_trace, y_trace, spikes = neuron.simulate_complete(n_steps, current, backend=backend)
    return x_trace, y_trace, spikes, neuron.x, neuron.y


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
        np.testing.assert_allclose(observed[1], reference[1], atol=tolerance, rtol=0.0)
        assert observed[2] == reference[2]
        assert observed[3] == pytest.approx(reference[3], abs=tolerance)
        assert observed[4] == pytest.approx(reference[4], abs=tolerance)


def _assert_enrolled_event_counts_and_trace_envelope(backend: str) -> None:
    for current in _OPERATING_CURRENTS:
        reference = _run("python", current=current)
        observed = _run(backend, current=current)
        tolerance = _TRACE_TOLERANCE[backend]
        np.testing.assert_allclose(observed[1], reference[1], atol=tolerance, rtol=0.0)
        assert observed[2] == reference[2]
        np.testing.assert_allclose(observed[0], reference[0], atol=tolerance, rtol=0.0)
        assert observed[3] == pytest.approx(reference[3], abs=tolerance)
        assert observed[4] == pytest.approx(reference[4], abs=tolerance)


def _assert_empty_and_single_step_preserve_state_contract(backend: str) -> None:
    for n_steps in (0, 1):
        reference = _run("python", n_steps=n_steps, current=0.01, x=0.2, y=0.7)
        observed = _run(backend, n_steps=n_steps, current=0.01, x=0.2, y=0.7)
        tolerance = _STEP_TOLERANCE[backend]
        np.testing.assert_allclose(observed[0], reference[0], atol=tolerance, rtol=0.0)
        np.testing.assert_allclose(observed[1], reference[1], atol=tolerance, rtol=0.0)
        assert observed[2] == reference[2]
        assert observed[3] == pytest.approx(reference[3], abs=tolerance)
        assert observed[4] == pytest.approx(reference[4], abs=tolerance)


def _assert_backend_contract(backend: str, available: Availability) -> None:
    _require(backend, available)
    _assert_source_equation_one_step_envelope(backend)
    _assert_enrolled_event_counts_and_trace_envelope(backend)
    _assert_empty_and_single_step_preserve_state_contract(backend)


__all__ = [
    "ctypes",
    "importlib",
    "os",
    "Callable",
    "np",
    "npt",
    "pytest",
    "chialvo_map",
    "ChialvoMapNeuron",
    "Availability",
    "RunResult",
    "_rust_available",
    "_julia_available",
    "_go_available",
    "_mojo_available",
    "_TRACE_TOLERANCE",
    "_STEP_TOLERANCE",
    "_OPERATING_CURRENTS",
    "_require",
    "_run",
    "_assert_source_equation_one_step_envelope",
    "_assert_enrolled_event_counts_and_trace_envelope",
    "_assert_empty_and_single_step_preserve_state_contract",
    "_assert_backend_contract",
]
