# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured five-runtime source APSDM contracts

"""Exercise all actual APSDM backends over the same complete configurations."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel.sigma_delta import (
    PARITY_ATOL,
    backend_available,
    simulate_sigma_delta,
)
from tests.test_sigma_delta_engine_binding_configuration import (
    PROFILES,
    _drive,
    _parameters,
)


@pytest.mark.parametrize("overrides", PROFILES)
@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_configured_public_backend_preserves_complete_trace(
    overrides: dict[str, float],
    backend: str,
) -> None:
    """Exercise all five actual configured runtime routes with exact event traces."""
    assert backend_available(backend), f"required configured APSDM backend {backend} unavailable"
    parameters = _parameters(overrides)
    expected = simulate_sigma_delta(_drive(), **parameters, backend="python")
    actual = simulate_sigma_delta(_drive(), **parameters, backend=backend)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    for key in ("sigma", "reconstruction"):
        output = np.asarray(actual[key])
        assert output.dtype == np.float64 and output.shape == (256,)
        np.testing.assert_allclose(
            output, np.asarray(expected[key], dtype=np.float64), rtol=0.0, atol=PARITY_ATOL[backend]
        )
    assert np.asarray(actual["events"]).dtype == np.int64
    for key in ("sigma_final", "reconstruction_final"):
        assert actual[key] == pytest.approx(expected[key], rel=0.0, abs=PARITY_ATOL[backend])
