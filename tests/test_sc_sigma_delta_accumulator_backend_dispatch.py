# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
import numpy as np
import pytest
from sc_neurocore.accel.sc_sigma_delta_accumulator import (
    PARITY_ATOL,
    backend_available,
    simulate_sc_sigma_delta_accumulator,
)


@pytest.mark.parametrize("backend", tuple(PARITY_ATOL))
def test_sc_sigma_delta_backend_matches_python(backend: str) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    drive = np.array([0.0] * 8 + [3.25, -4.5, 0.2] * 32)
    expected = simulate_sc_sigma_delta_accumulator(drive, backend="python")
    actual = simulate_sc_sigma_delta_accumulator(drive, backend=backend)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    np.testing.assert_array_equal(actual["sigma"], expected["sigma"])
