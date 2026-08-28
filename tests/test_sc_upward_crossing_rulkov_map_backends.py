# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained Rulkov polyglot parity contracts

"""Five-runtime parity for the retained upward-crossing identity."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import sc_upward_crossing_rulkov_map as implementation
from sc_neurocore.neurons.models.sc_upward_crossing_rulkov_map import (
    SCUpwardCrossingRulkovMapNeuron,
)


@pytest.mark.parametrize("backend", ["rust", "julia", "go", "mojo"])
@pytest.mark.parametrize(("current", "threshold"), [(0.0, 0.0), (0.5, 0.25), (2.0, 0.0)])
def test_complete_trace_and_events_match_python(
    backend: str, current: float, threshold: float
) -> None:
    """Every compiled runtime must preserve state and retained event timing."""
    if not implementation._backend_available(backend):
        pytest.skip(f"{backend} retained Rulkov backend unavailable")
    reference = SCUpwardCrossingRulkovMapNeuron(x_threshold=threshold)
    candidate = SCUpwardCrossingRulkovMapNeuron(x_threshold=threshold)

    expected_trace, expected_events = reference.simulate(2048, current, backend="python")
    actual_trace, actual_events = candidate.simulate(2048, current, backend=backend)

    tolerance = 1.0e-9 if backend == "mojo" else 0.0
    np.testing.assert_allclose(actual_trace, expected_trace, atol=tolerance, rtol=0.0)
    assert actual_events == expected_events
    assert candidate.x == pytest.approx(reference.x, abs=tolerance, rel=0.0)
    assert candidate.y == pytest.approx(reference.y, abs=tolerance, rel=0.0)


@pytest.mark.parametrize("backend", ["rust", "julia", "go", "mojo"])
def test_compiled_batch_rejects_non_finite_candidate_atomically(backend: str) -> None:
    """Compiled backends must reject overflow without committing partial state."""
    if not implementation._backend_available(backend):
        pytest.skip(f"{backend} retained Rulkov backend unavailable")
    neuron = SCUpwardCrossingRulkovMapNeuron(x=0.5, y=1.0e308, alpha=1.0e308)
    before = (neuron.x, neuron.y)

    with pytest.raises(FloatingPointError):
        neuron.simulate(2, 0.0, backend=backend)

    assert (neuron.x, neuron.y) == before


def test_auto_dispatch_matches_the_public_python_floor() -> None:
    """Automatic dispatch must preserve the retained public contract."""
    expected_model = SCUpwardCrossingRulkovMapNeuron(x_threshold=0.25)
    actual_model = SCUpwardCrossingRulkovMapNeuron(x_threshold=0.25)
    expected = expected_model.simulate(1024, 0.5, backend="python")
    actual = actual_model.simulate(1024, 0.5, backend="auto")

    np.testing.assert_allclose(actual[0], expected[0], atol=1.0e-9, rtol=0.0)
    assert actual[1] == expected[1]


@pytest.mark.parametrize(
    ("backend", "runtime_attribute"),
    [
        ("rust", "_rust_simulate"),
        ("julia", "_julia_module"),
        ("go", "_go_lib"),
        ("mojo", "_mojo_lib"),
    ],
)
def test_runtime_disappearance_fails_closed_without_state_mutation(
    monkeypatch: pytest.MonkeyPatch, backend: str, runtime_attribute: str
) -> None:
    """A retained runtime lost after selection must not commit public state."""

    def report_available(_backend: str) -> bool:
        return True

    monkeypatch.setattr(implementation, "_backend_available", report_available)
    monkeypatch.setattr(implementation, runtime_attribute, None)
    neuron = SCUpwardCrossingRulkovMapNeuron()
    before = (neuron.x, neuron.y)

    message = f"{backend} SC upward-crossing Rulkov backend is unavailable"
    with pytest.raises(RuntimeError, match=message):
        neuron.simulate(1, 0.5, backend=backend)

    assert (neuron.x, neuron.y) == before
