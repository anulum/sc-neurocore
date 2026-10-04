# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured five-runtime retained non-resetting adaptive LIF contracts

"""Compare complete configured exact-flow trajectories on five real runtimes."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from sc_neurocore.accel.sc_non_resetting_adaptive_lif import (
    PARITY_ATOL,
    backend_available,
    simulate_sc_non_resetting_adaptive_lif,
)
from tests.test_sc_non_resetting_adaptive_lif_engine_binding_configuration import (
    PROFILES,
    _drive,
    _parameters,
)


@pytest.mark.parametrize("overrides", PROFILES)
@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_configured_public_backend_preserves_trace_and_empty_state(
    overrides: dict[str, float], backend: str
) -> None:
    """Preserve full voltage/threshold traces, exact events and empty finals."""
    assert backend_available(backend), (
        f"required configured SC non-resetting adaptive LIF backend {backend} unavailable"
    )
    parameters = _parameters(overrides)
    expected = simulate_sc_non_resetting_adaptive_lif(_drive(), backend="python", **parameters)
    actual = simulate_sc_non_resetting_adaptive_lif(_drive(), backend=backend, **parameters)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    assert np.asarray(actual["events"]).dtype == np.int64
    for key in ("voltages", "theta"):
        output = np.asarray(actual[key])
        assert output.dtype == np.float64 and output.shape == (256,)
        np.testing.assert_allclose(
            output, np.asarray(expected[key]), rtol=0, atol=PARITY_ATOL[backend]
        )
    for key in ("v_final", "theta_final"):
        assert float(cast(float, actual[key])) == pytest.approx(
            expected[key], rel=0, abs=PARITY_ATOL[backend]
        )
    empty = simulate_sc_non_resetting_adaptive_lif([], backend=backend, **parameters)
    for key in ("voltages", "theta", "events"):
        assert np.asarray(empty[key]).shape == (0,)
    assert empty["v_final"] == parameters["v"] and empty["theta_final"] == parameters["theta"]
