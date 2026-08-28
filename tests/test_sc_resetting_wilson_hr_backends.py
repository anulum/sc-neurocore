# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained resetting Wilson-HR polyglot parity

"""Real-runtime parity for the retained resetting Wilson-HR recurrence."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from sc_neurocore.neurons.models import sc_resetting_wilson_hr as implementation
from sc_neurocore.neurons.models.sc_resetting_wilson_hr import SCResettingWilsonHRNeuron

BackendAvailable = Callable[[], bool]
RunResult = tuple[NDArray[np.float64], int, float, float]


def _run(
    backend: str,
    *,
    n_steps: int = 4_000,
    current: float = 2.0,
) -> RunResult:
    neuron = SCResettingWilsonHRNeuron()
    trace, events = neuron.simulate(n_steps, current, backend=backend)
    return trace, events, neuron.v, neuron.r


def _rust_available() -> bool:
    return implementation._HAS_RUST


def _julia_available() -> bool:
    return implementation._ensure_julia_loaded()


def _go_available() -> bool:
    return implementation._ensure_go_loaded()


def _mojo_available() -> bool:
    return implementation._ensure_mojo_loaded()


_EXACT_BACKENDS: tuple[tuple[str, BackendAvailable], ...] = (
    ("rust", _rust_available),
    ("julia", _julia_available),
    ("go", _go_available),
)
_ALL_BACKENDS: tuple[tuple[str, BackendAvailable], ...] = (
    *_EXACT_BACKENDS,
    ("mojo", _mojo_available),
)


@pytest.mark.parametrize("backend,available", _EXACT_BACKENDS, ids=lambda value: str(value))
@pytest.mark.parametrize("current", (0.0, 2.0, 10.0))
def test_exact_backends_match_complete_python_trajectory(
    backend: str, available: BackendAvailable, current: float
) -> None:
    if not available():
        pytest.skip(f"{backend} SC resetting Wilson-HR backend unavailable")
    expected = _run("python", current=current)
    actual = _run(backend, current=current)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


@pytest.mark.parametrize("backend,available", _ALL_BACKENDS, ids=lambda value: str(value))
@pytest.mark.parametrize("n_steps", (0, 1, 2))
def test_every_backend_preserves_empty_and_initial_update_contracts(
    backend: str, available: BackendAvailable, n_steps: int
) -> None:
    if not available():
        pytest.skip(f"{backend} SC resetting Wilson-HR backend unavailable")
    expected = _run("python", n_steps=n_steps)
    actual = _run(backend, n_steps=n_steps)
    np.testing.assert_allclose(actual[0], expected[0], atol=2.5e-14, rtol=0.0)
    assert actual[1] == expected[1]
    assert abs(actual[2] - expected[2]) <= 2.5e-14
    assert abs(actual[3] - expected[3]) <= 2.5e-14


@pytest.mark.parametrize("current", (0.0, 2.0, 10.0))
def test_mojo_matches_complete_trajectory_and_event_count(current: float) -> None:
    if not _mojo_available():
        pytest.skip("Mojo SC resetting Wilson-HR backend unavailable")
    expected = _run("python", current=current)
    actual = _run("mojo", current=current)
    np.testing.assert_allclose(actual[0], expected[0], atol=2.5e-12, rtol=0.0)
    assert actual[1] == expected[1]
    assert abs(actual[2] - expected[2]) <= 2.5e-12
    assert abs(actual[3] - expected[3]) <= 2.5e-12


def test_auto_dispatch_matches_python_reference() -> None:
    expected = _run("python")
    actual = _run("auto")
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


@pytest.mark.parametrize("backend,available", _ALL_BACKENDS, ids=lambda value: str(value))
def test_native_batch_rejection_is_failure_atomic(
    backend: str, available: BackendAvailable
) -> None:
    if not available():
        pytest.skip(f"{backend} SC resetting Wilson-HR backend unavailable")
    neuron = SCResettingWilsonHRNeuron(v=1.0e103)
    before = (neuron.v, neuron.r)
    with pytest.raises(FloatingPointError, match="invalid|rejected"):
        neuron.simulate(2, 2.0, backend=backend)
    assert (neuron.v, neuron.r) == before
