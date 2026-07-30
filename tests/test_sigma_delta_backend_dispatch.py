# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
import numpy as np
import pytest
from sc_neurocore.accel.sigma_delta import PARITY_ATOL, backend_available, simulate_sigma_delta


@pytest.mark.parametrize("backend", tuple(PARITY_ATOL))
def test_sigma_delta_backend_matches_python(backend: str) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    drive = np.array([0.0] * 8 + [2.0, -1.0, 4.0] * 32)
    expected = simulate_sigma_delta(drive, backend="python")
    actual = simulate_sigma_delta(drive, backend=backend)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    np.testing.assert_allclose(
        actual["sigma"], expected["sigma"], atol=PARITY_ATOL[backend], rtol=0
    )
    np.testing.assert_allclose(
        actual["reconstruction"], expected["reconstruction"], atol=PARITY_ATOL[backend], rtol=0
    )
