# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
import numpy as np
import pytest
from sc_neurocore.accel.sc_normalized_energy_lif import (
    PARITY_ATOL,
    backend_available,
    simulate_sc_normalized_energy_lif,
)


@pytest.mark.parametrize("backend", tuple(PARITY_ATOL))
def test_sc_energy_lif_backend_matches_python(backend: str) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    drive = np.resize(np.array([30.0, 0.0, 50.0, 10.0]), 128)
    expected = simulate_sc_normalized_energy_lif(drive, backend="python")
    actual = simulate_sc_normalized_energy_lif(drive, backend=backend)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    for key in ("voltages", "epsilon"):
        np.testing.assert_allclose(actual[key], expected[key], atol=PARITY_ATOL[backend], rtol=0)
