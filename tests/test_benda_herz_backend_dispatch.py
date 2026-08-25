# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from __future__ import annotations
import numpy as np
import pytest
from sc_neurocore.accel.benda_herz import PARITY_ATOL, backend_available, simulate_benda_herz

DRIVE = np.tile(np.array([0.0, 1.0, 4.0, 9.0]), 64)


def test_python_reference_receipt() -> None:
    result = simulate_benda_herz(np.tile(DRIVE, 2), backend="python")
    assert int(np.sum(result["events"])) == 3
    assert result["a_final"] == pytest.approx(2.6220722275910986, abs=1e-14)
    assert result["phase_final"] == pytest.approx(0.38422886217335506, abs=1e-14)


@pytest.mark.parametrize("backend", ["rust", "julia", "go", "mojo"])
def test_native_backend_matches_python(backend: str) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    expected = simulate_benda_herz(DRIVE, backend="python")
    actual = simulate_benda_herz(DRIVE, backend=backend)
    np.testing.assert_allclose(
        actual["adaptation"], expected["adaptation"], rtol=0, atol=PARITY_ATOL[backend]
    )
    np.testing.assert_allclose(
        actual["phases"], expected["phases"], rtol=0, atol=PARITY_ATOL[backend]
    )
    np.testing.assert_array_equal(actual["events"], expected["events"])


def test_unknown_backend_fails() -> None:
    with pytest.raises(ValueError, match="unknown"):
        simulate_benda_herz([1.0], backend="cuda")
