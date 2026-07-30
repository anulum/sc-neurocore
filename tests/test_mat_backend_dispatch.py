# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - source MAT* backend dispatch parity

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel.mat import PARITY_ATOL, backend_available, simulate_mat


@pytest.mark.parametrize("backend", list(PARITY_ATOL))
def test_mat_backend_matches_complete_python_trace(backend: str) -> None:
    """Every installed runtime preserves all source MAT* state/event traces."""
    if not backend_available(backend):
        pytest.skip(f"{backend} MAT backend unavailable")
    currents = np.concatenate(
        (np.zeros(32), np.full(5000, 0.5), np.tile(np.array([0.2, 0.7]), 512))
    )
    expected = simulate_mat(currents, backend="python")
    actual = simulate_mat(currents, backend=backend)
    for key in ("voltages", "theta1", "theta2", "refractory"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=PARITY_ATOL[backend])
    np.testing.assert_array_equal(actual["events"], expected["events"])


def test_mat_dispatch_rejects_invalid_input_without_fallback() -> None:
    with pytest.raises(ValueError, match="finite"):
        simulate_mat([0.0, float("nan")], backend="python")
