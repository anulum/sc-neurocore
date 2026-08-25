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
from sc_neurocore.accel.sc_stochastic_rate_adaptation import (
    PARITY_ATOL,
    backend_available,
    simulate_sc_stochastic_rate_adaptation,
)

DRIVE = np.tile(np.array([0.0, 10.0, 25.0, 50.0]), 64)
UNIFORMS = np.random.default_rng(42).random(DRIVE.size)


@pytest.mark.parametrize("backend", ["rust", "julia", "go", "mojo"])
def test_backend_matches_controlled_python(backend: str) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    expected = simulate_sc_stochastic_rate_adaptation(DRIVE, UNIFORMS, backend="python")
    actual = simulate_sc_stochastic_rate_adaptation(DRIVE, UNIFORMS, backend=backend)
    np.testing.assert_allclose(
        actual["adaptation"], expected["adaptation"], rtol=0, atol=PARITY_ATOL[backend]
    )
    np.testing.assert_array_equal(actual["events"], expected["events"])


def test_uniform_validation() -> None:
    with pytest.raises(ValueError):
        simulate_sc_stochastic_rate_adaptation([1.0], [1.0])
