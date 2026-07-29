# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Nagumo–Sato five-runtime parity

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel import nagumo_sato_map

_BACKENDS = ("rust", "julia", "go", "mojo")


def _drive() -> np.ndarray:
    index = np.arange(1024, dtype=np.float64)
    return 0.05 * np.sin(index * 0.037) - 0.02 * np.cos(index * 0.011)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_complete_state_and_event_trace_parity(backend: str) -> None:
    if not nagumo_sato_map.backend_available(backend):
        pytest.skip(f"{backend} Nagumo-Sato backend unavailable")
    expected = nagumo_sato_map.simulate_nagumo_sato_map(current=_drive(), backend="python")
    observed = nagumo_sato_map.simulate_nagumo_sato_map(current=_drive(), backend=backend)
    np.testing.assert_allclose(
        observed["y"], expected["y"], rtol=0.0, atol=nagumo_sato_map.PARITY_ATOL[backend]
    )
    for key in ("x", "spikes"):
        np.testing.assert_array_equal(observed[key], expected[key])
    assert observed["y_final"] == pytest.approx(
        expected["y_final"], abs=nagumo_sato_map.PARITY_ATOL[backend]
    )
    assert observed["x_final"] == expected["x_final"]
    assert observed["spike_count"] == expected["spike_count"]


def test_result_guard_rejects_non_source_events() -> None:
    result = nagumo_sato_map.simulate_nagumo_sato_map(current=[0.0, 0.0], backend="python")
    forged = dict(result)
    forged["spikes"] = np.asarray([1.0, 0.0])
    forged["spike_count"] = 1
    with pytest.raises(FloatingPointError, match=r"H\(y"):
        nagumo_sato_map.normalise_result(forged, n_steps=2, initial_y=0.1)
