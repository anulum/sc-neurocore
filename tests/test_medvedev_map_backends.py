# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev 2005 first-return polyglot parity

"""Cross-language parity for the calibrated slow-calcium recurrence."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
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
) -> tuple[np.ndarray, int, float]:
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


@pytest.mark.parametrize("backend", ("python", "rust", "julia", "go", "mojo"))
def test_invalid_current_fails_before_backend_mutation(backend: str) -> None:
    """Front-door validation rejects NaN before any backend writes state."""
    neuron = MedvedevMapNeuron()
    before = neuron.u
    with pytest.raises(ValueError, match="current must be finite"):
        neuron.simulate(10, float("nan"), backend=backend)
    assert neuron.u == before
