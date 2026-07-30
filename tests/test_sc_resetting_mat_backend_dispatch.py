# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - SC resetting-MAT backend dispatch parity

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel.sc_resetting_mat import (
    PARITY_ATOL,
    backend_available,
    simulate_sc_resetting_mat,
)


@pytest.mark.parametrize("backend", list(PARITY_ATOL))
def test_sc_resetting_mat_backend_matches_historical_trace(backend: str) -> None:
    """Every installed runtime preserves the complete audited SC trace."""
    if not backend_available(backend):
        pytest.skip(f"{backend} SC resetting-MAT backend unavailable")
    currents = np.array([0.0] * 32 + [50.0] * 96 + [20.0, 60.0] * 64)
    expected = simulate_sc_resetting_mat(currents, backend="python")
    actual = simulate_sc_resetting_mat(currents, backend=backend)
    for key in ("voltages", "theta1", "theta2"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=PARITY_ATOL[backend])
    np.testing.assert_array_equal(actual["events"], expected["events"])


def test_sc_resetting_mat_dispatch_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown"):
        simulate_sc_resetting_mat([0.0], backend="surrogate")
