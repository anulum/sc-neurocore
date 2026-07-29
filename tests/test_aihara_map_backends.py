# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aihara polyglot parity contracts

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel import aihara_map
from sc_neurocore.neurons.models.aihara_map_neuron import AiharaMapNeuron

_BACKENDS = ("rust", "julia", "go", "mojo")


def _drive() -> np.ndarray:
    index = np.arange(512, dtype=np.float64)
    return 0.04 * np.sin(index * 0.037) - 0.02 * np.cos(index * 0.011)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_complete_trace_parity(backend: str) -> None:
    if not aihara_map.backend_available(backend):
        pytest.skip(f"{backend} Aihara backend unavailable")
    expected = aihara_map.simulate_aihara_map(current=_drive(), backend="python")
    observed = aihara_map.simulate_aihara_map(current=_drive(), backend=backend)
    tolerance = aihara_map.PARITY_ATOL[backend]
    for key in ("y", "x"):
        np.testing.assert_allclose(observed[key], expected[key], rtol=0.0, atol=tolerance)
    np.testing.assert_array_equal(observed["spikes"], expected["spikes"])
    assert observed["spike_count"] == expected["spike_count"]
    assert observed["y_final"] == pytest.approx(expected["y_final"], abs=tolerance)
    assert observed["x_final"] == pytest.approx(expected["x_final"], abs=tolerance)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_short_horizon_equation_parity_is_tight(backend: str) -> None:
    """Before chaotic amplification, all active lanes agree to binary64 scale."""
    if not aihara_map.backend_available(backend):
        pytest.skip(f"{backend} Aihara backend unavailable")
    drive = _drive()[:64]
    expected = aihara_map.simulate_aihara_map(current=drive, backend="python")
    observed = aihara_map.simulate_aihara_map(current=drive, backend=backend)
    np.testing.assert_allclose(observed["y"], expected["y"], rtol=0.0, atol=5.0e-11)
    np.testing.assert_array_equal(observed["spikes"], expected["spikes"])


def test_empty_and_single_step_receipts() -> None:
    empty = aihara_map.simulate_aihara_map(current=[], backend="python")
    assert empty["y_final"] == 0.1
    assert empty["spike_count"] == 0
    single = aihara_map.simulate_aihara_map(current=[0.0], backend="python")
    neuron = AiharaMapNeuron()
    expected_event = neuron.step(0.0)
    assert single["y_final"] == neuron.y
    assert single["spike_count"] == expected_event


def test_explicit_unavailable_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aihara_map, "_HAS_RUST", False)
    monkeypatch.setattr(aihara_map, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="Rust aihara backend is unavailable"):
        aihara_map.simulate_aihara_map(current=[0.0], backend="rust")


def test_result_guard_rejects_crossing_semantics() -> None:
    result = aihara_map.simulate_aihara_map(
        y=-0.1, k=0.0, alpha=0.01, bias=0.2, current=[0.0, 0.0], backend="python"
    )
    forged = dict(result)
    forged["spikes"] = np.array([1.0, 0.0])
    forged["spike_count"] = 1
    with pytest.raises(FloatingPointError, match="waveform shaper"):
        aihara_map.normalise_result(forged, n_steps=2, initial_y=-0.1, epsilon=0.01)
