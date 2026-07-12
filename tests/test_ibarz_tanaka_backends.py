# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka 2007 polyglot parity tests

"""Cross-backend parity for the source-derived four-branch map."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sc_neurocore.neurons.models import ibarz_tanaka_map
from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron


def _run(
    backend: str, *, current: float = 0.2, n_steps: int = 1000
) -> tuple[np.ndarray, int, float, float]:
    neuron = IbarzTanakaMapNeuron()
    trace, events = neuron.simulate(n_steps, current, backend=backend)
    return trace, events, neuron.v, neuron.u


def _rust() -> bool:
    return ibarz_tanaka_map._HAS_RUST


def _julia() -> bool:
    return ibarz_tanaka_map._ensure_julia_loaded()


def _go() -> bool:
    return ibarz_tanaka_map._ensure_go_loaded()


def _mojo() -> bool:
    return ibarz_tanaka_map._ensure_mojo_loaded()


_BIT_EXACT: tuple[tuple[str, Callable[[], bool]], ...] = (
    ("rust", _rust),
    ("julia", _julia),
    ("go", _go),
)


@pytest.mark.parametrize(("backend", "available"), _BIT_EXACT)
@pytest.mark.parametrize("current", (0.0, 0.2, 1.0))
def test_exact_backends_match_python(
    backend: str, available: Callable[[], bool], current: float
) -> None:
    """Rust, Julia and Go reproduce every committed fast-state bit."""
    if not available():
        pytest.skip(f"{backend} backend is not built")
    expected = _run("python", current=current)
    actual = _run(backend, current=current)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


@pytest.mark.parametrize("current", (0.0, 0.2, 1.0))
def test_mojo_matches_source_events_and_measured_error_bound(current: float) -> None:
    """Mojo preserves events and stays within the measured absolute-error bound."""
    if not _mojo():
        pytest.skip("mojo backend is not built")
    expected = _run("python", current=current)
    actual = _run("mojo", current=current)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1.5e-8)
    assert actual[1] == expected[1]
    assert actual[2] == pytest.approx(expected[2], abs=1.5e-8)
    assert actual[3] == pytest.approx(expected[3], abs=1.5e-8)


def test_zero_step_contract_preserves_state_across_available_backends() -> None:
    """The empty batch returns an empty trace and preserves both states."""
    backends = [("python", lambda: True), *_BIT_EXACT, ("mojo", _mojo)]
    for backend, available in backends:
        if not available():
            continue
        trace, events, v_final, u_final = _run(backend, n_steps=0)
        assert trace.size == 0
        assert events == 0
        assert (v_final, u_final) == (-1.0, -0.1)


def test_explicit_unavailable_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit unavailable backend cannot silently fall back to Python."""
    monkeypatch.setattr(ibarz_tanaka_map, "_HAS_RUST", False)
    with pytest.raises(RuntimeError, match="rust Ibarz-Tanaka backend is unavailable"):
        IbarzTanakaMapNeuron().simulate(1, 0.2, backend="rust")
