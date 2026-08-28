# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GLIF5 polyglot parity contracts

"""Complete-state parity through every public GLIF5 batch backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from sc_neurocore.neurons.models import glif
from sc_neurocore.neurons.models.glif import GLIFNeuron


def _run(
    backend: str, *, n: int = 4096, current: float = 30.0, **params: float
) -> tuple[NDArray[np.float64], int, tuple[float, float, float, float, float, float]]:
    neuron = GLIFNeuron(**params)
    trace, events = neuron.simulate(n, current, backend=backend)
    state = (
        neuron.v,
        neuron.theta_spike,
        neuron.i_asc1,
        neuron.i_asc2,
        neuron.theta_voltage,
        neuron.refractory_remaining,
    )
    return trace, events, state


_BACKENDS: tuple[tuple[str, Callable[[], bool], float], ...] = (
    ("rust", lambda: glif._HAS_RUST, 0.0),
    ("julia", glif._ensure_julia_loaded, 2e-12),
    ("go", glif._ensure_go_loaded, 2e-12),
    ("mojo", glif._ensure_mojo_loaded, 2e-12),
)
_REGIMES: tuple[dict[str, float], ...] = (
    {},
    {"a_voltage": 0.0, "delta_theta_spike": 0.0},
    {"f_v": 0.35, "delta_v": 1.25, "refractory_period": 0.0},
    {"k_asc1": 0.2, "k_asc2": 0.02, "f_asc1": 0.6, "f_asc2": -0.2},
)


@pytest.mark.parametrize("backend,available,tolerance", _BACKENDS)
@pytest.mark.parametrize("current", (0.0, 22.0, 30.0, 50.0))
def test_complete_trace_and_state_parity(
    backend: str, available: Callable[[], bool], tolerance: float, current: float
) -> None:
    if not available():
        pytest.skip(f"{backend} GLIF5 backend unavailable")
    expected = _run("python", current=current)
    actual = _run(backend, current=current)

    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=tolerance)
    assert actual[1] == expected[1]
    np.testing.assert_allclose(actual[2], expected[2], rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("backend,available,tolerance", _BACKENDS)
@pytest.mark.parametrize("parameters", _REGIMES)
def test_parameterized_reset_and_threshold_parity(
    backend: str,
    available: Callable[[], bool],
    tolerance: float,
    parameters: dict[str, float],
) -> None:
    if not available():
        pytest.skip(f"{backend} GLIF5 backend unavailable")
    expected = _run("python", n=1024, current=35.0, **parameters)
    actual = _run(backend, n=1024, current=35.0, **parameters)

    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=tolerance)
    assert actual[1] == expected[1]
    np.testing.assert_allclose(actual[2], expected[2], rtol=0.0, atol=tolerance)


def test_auto_backend_matches_reference_contract() -> None:
    expected = _run("python")
    actual = _run("auto")

    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=2e-12)
    assert actual[1:] == expected[1:]


@pytest.mark.parametrize("n_steps", [-1, True, 1.5])
def test_invalid_step_count_is_rejected(n_steps: object) -> None:
    simulate = cast(Callable[[object, float], object], GLIFNeuron().simulate)
    with pytest.raises(ValueError, match="n_steps"):
        simulate(n_steps, 0.0)


def test_invalid_backend_and_current_are_rejected() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        GLIFNeuron().simulate(1, 0.0, backend="cuda")
    with pytest.raises(ValueError, match="current must be finite"):
        GLIFNeuron().simulate(1, np.inf)


def test_explicit_unavailable_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glif, "_HAS_RUST", False)
    with pytest.raises(RuntimeError, match="rust GLIF5 backend is unavailable"):
        GLIFNeuron().simulate(1, 0.0, backend="rust")
