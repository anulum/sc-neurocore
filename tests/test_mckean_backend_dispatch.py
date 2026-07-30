# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import numpy as np
import pytest

from sc_neurocore.accel.mckean import PARITY_ATOL, backend_available, simulate_mckean
from sc_neurocore.accel.sc_triangular_mckean import (
    PARITY_ATOL as SC_PARITY_ATOL,
)
from sc_neurocore.accel.sc_triangular_mckean import (
    backend_available as sc_backend_available,
)
from sc_neurocore.accel.sc_triangular_mckean import simulate_sc_triangular_mckean


@pytest.mark.parametrize("backend", tuple(PARITY_ATOL))
def test_source_backend_matches_python(backend: str) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    drive = np.resize(np.array([0.0, 3.0, 0.0, -0.2]), 128)
    expected = simulate_mckean(drive, backend="python")
    actual = simulate_mckean(drive, backend=backend)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    for key in ("voltages", "recovery"):
        np.testing.assert_allclose(actual[key], expected[key], atol=PARITY_ATOL[backend], rtol=0)


@pytest.mark.parametrize("backend", tuple(SC_PARITY_ATOL))
def test_sc_backend_matches_python(backend: str) -> None:
    if not sc_backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    drive = np.resize(np.array([0.3, 0.5, 0.8, 0.1]), 64)
    expected = simulate_sc_triangular_mckean(drive, backend="python")
    actual = simulate_sc_triangular_mckean(drive, backend=backend)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    for key in ("voltages", "recovery"):
        np.testing.assert_allclose(actual[key], expected[key], atol=SC_PARITY_ATOL[backend], rtol=0)


@pytest.mark.parametrize("simulate", [simulate_mckean, simulate_sc_triangular_mckean])
def test_dispatch_rejects_nonfinite_and_unknown_backends(simulate) -> None:
    with pytest.raises(ValueError, match="finite and one-dimensional"):
        simulate([0.0, np.nan])
    with pytest.raises(ValueError, match="unknown"):
        simulate([0.0], backend="cuda")
