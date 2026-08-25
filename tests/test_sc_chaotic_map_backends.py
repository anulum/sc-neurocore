# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel import sc_chaotic_map
from sc_neurocore.neurons.models import SCChaoticMapNeuron

_BACKENDS = ("rust", "julia", "go", "mojo")


def _drive() -> np.ndarray:
    index = np.arange(512, dtype=np.float64)
    return 0.15 * np.sin(index * 0.037) - 0.05 * np.cos(index * 0.011)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_complete_trace_parity(backend: str) -> None:
    if not sc_chaotic_map.backend_available(backend):
        pytest.skip(f"{backend} SC chaotic-map backend unavailable")
    expected = sc_chaotic_map.simulate_sc_chaotic_map(0.4, -0.2, current=_drive(), backend="python")
    observed = sc_chaotic_map.simulate_sc_chaotic_map(0.4, -0.2, current=_drive(), backend=backend)
    tolerance = sc_chaotic_map.PARITY_ATOL[backend]
    for key in ("x", "y"):
        np.testing.assert_allclose(observed[key], expected[key], rtol=0.0, atol=tolerance)
    np.testing.assert_array_equal(observed["spikes"], expected["spikes"])
    assert observed["spike_count"] == expected["spike_count"]
    assert observed["x_final"] == pytest.approx(expected["x_final"], abs=tolerance)
    assert observed["y_final"] == pytest.approx(expected["y_final"], abs=tolerance)


def test_model_batch_commits_final_state() -> None:
    neuron = SCChaoticMapNeuron(x=0.4, y=-0.2)
    result = neuron.simulate([0.1, 0.1], backend="python")
    assert result["spike_count"] == 1
    assert (neuron.x, neuron.y) == (result["x_final"], result["y_final"])


def test_empty_receipt_preserves_initial_state() -> None:
    result = sc_chaotic_map.simulate_sc_chaotic_map(0.4, -0.2, current=[], backend="python")
    assert result["x_final"] == 0.4
    assert result["y_final"] == -0.2
    assert result["spike_count"] == 0


def test_result_guard_rejects_level_events() -> None:
    result = sc_chaotic_map.simulate_sc_chaotic_map(0.4, -0.2, current=[0.1, 0.1], backend="python")
    forged = dict(result)
    forged["spikes"] = np.array([1.0, 1.0])
    forged["spike_count"] = 2
    with pytest.raises(FloatingPointError, match="upward-crossing"):
        sc_chaotic_map.normalise_result(
            forged, n_steps=2, initial_x=0.4, initial_y=-0.2, threshold=0.5
        )


def test_explicit_unavailable_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc_chaotic_map, "_HAS_RUST", False)
    monkeypatch.setattr(sc_chaotic_map, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="Rust SC chaotic-map backend is unavailable"):
        sc_chaotic_map.simulate_sc_chaotic_map(current=[0.0], backend="rust")
