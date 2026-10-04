# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured five-runtime source MAT(1) contracts

"""Exercise all actual MAT(1) backends over the same complete configurations."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel.non_resetting_lif import (
    PARITY_ATOL,
    backend_available,
    simulate_non_resetting_lif,
)
from tests.test_non_resetting_lif_engine_binding_configuration import (
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
    assert backend_available(backend), f"required configured MAT(1) backend {backend} unavailable"
    parameters = _parameters(overrides)
    expected = simulate_non_resetting_lif(_drive(), **parameters, backend="python")
    actual = simulate_non_resetting_lif(_drive(), **parameters, backend=backend)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    for key in ("voltages", "theta", "refractory"):
        output = np.asarray(actual[key])
        assert output.dtype == np.float64 and output.shape == (256,)
        np.testing.assert_allclose(output, expected[key], rtol=0.0, atol=PARITY_ATOL[backend])
    assert np.asarray(actual["events"]).dtype == np.int64
    for key in ("v_final", "theta_final", "refractory_final"):
        assert actual[key] == pytest.approx(expected[key], rel=0.0, abs=PARITY_ATOL[backend])
