# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC adaptive-threshold-map five-runtime parity

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel import sc_adaptive_threshold_map

_BACKENDS = ("rust", "julia", "go", "mojo")


def _drive() -> np.ndarray:
    index = np.arange(1024, dtype=np.float64)
    return 0.6 + 0.25 * np.sin(index * 0.017) - 0.05 * np.cos(index * 0.031)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_complete_state_and_event_trace_parity(backend: str) -> None:
    if not sc_adaptive_threshold_map.backend_available(backend):
        pytest.skip(f"{backend} SC adaptive-map backend unavailable")
    expected = sc_adaptive_threshold_map.simulate_sc_adaptive_threshold_map(
        current=_drive(), backend="python"
    )
    observed = sc_adaptive_threshold_map.simulate_sc_adaptive_threshold_map(
        current=_drive(), backend=backend
    )
    tolerance = sc_adaptive_threshold_map.PARITY_ATOL[backend]
    for key in ("x", "theta"):
        np.testing.assert_allclose(observed[key], expected[key], rtol=0.0, atol=tolerance)
    np.testing.assert_array_equal(observed["spikes"], expected["spikes"])
    assert observed["spike_count"] == expected["spike_count"]


def test_result_guard_rejects_crossing_drift() -> None:
    result = sc_adaptive_threshold_map.simulate_sc_adaptive_threshold_map(
        current=[0.6, 0.6], backend="python"
    )
    forged = dict(result)
    forged["spikes"] = np.asarray([0.0, 1.0])
    forged["spike_count"] = 1
    with pytest.raises(FloatingPointError, match="upward-crossing"):
        sc_adaptive_threshold_map.normalise_result(
            forged, n_steps=2, initial_x=0.0, initial_theta=0.0, threshold=0.8
        )
