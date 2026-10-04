# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured SC resetting-MAT five-runtime contracts

"""Compare all configured SC RK4/reset state and event traces on real runtimes."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from sc_neurocore.accel.sc_resetting_mat import (
    PARITY_ATOL,
    backend_available,
    simulate_sc_resetting_mat,
)

PROFILE = {
    "v": -66.0,
    "theta1": 2.0,
    "theta2": 3.0,
    "v_rest": -70.0,
    "v_reset": -71.0,
    "v_threshold_base": -49.0,
    "tau_m": 12.0,
    "tau_1": 15.0,
    "tau_2": 250.0,
    "h1": 4.0,
    "h2": 2.0,
    "resistance": 1.25,
    "dt": 0.25,
}


@pytest.mark.parametrize("dt", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("current", [0.0, 20.0, 50.0])
@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_complete_profile_preserves_full_trace_and_empty_finals(
    dt: float, current: float, backend: str
) -> None:
    """Match configured full RK4 trajectories, exact events and all empty finals."""
    assert backend_available(backend), f"required SC resetting-MAT backend {backend} unavailable"
    parameters = PROFILE | {"dt": dt}
    drive = np.resize(np.array([current, 0.0, 50.0, 10.0], dtype=np.float64), 256)
    before = drive.copy()
    drive.setflags(write=False)
    expected = simulate_sc_resetting_mat(drive, backend="python", **parameters)
    actual = simulate_sc_resetting_mat(drive, backend=backend, **parameters)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    assert np.asarray(actual["events"]).dtype == np.int64
    for key in ("voltages", "theta1", "theta2"):
        output = np.asarray(actual[key])
        assert output.dtype == np.float64 and output.shape == (256,)
        np.testing.assert_allclose(output, expected[key], rtol=0, atol=PARITY_ATOL[backend])
    for key in ("v_final", "theta1_final", "theta2_final"):
        assert float(cast(float, actual[key])) == pytest.approx(
            expected[key], rel=0, abs=PARITY_ATOL[backend]
        )
    empty = simulate_sc_resetting_mat([], backend=backend, **parameters)
    for key in ("voltages", "theta1", "theta2", "events"):
        assert np.asarray(empty[key]).shape == (0,)
    for field in ("v", "theta1", "theta2"):
        assert empty[f"{field}_final"] == parameters[field]
    np.testing.assert_array_equal(drive, before)
    assert not drive.flags.writeable


@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_finite_cancellation_preserves_source_derivative_and_events(backend: str) -> None:
    """Large finite cancelling terms cannot introduce a backend-specific spike."""
    assert backend_available(backend), f"required SC resetting-MAT backend {backend} unavailable"
    result = simulate_sc_resetting_mat(
        np.full(4, -1.0), v_rest=1e308, resistance=1e308, backend=backend
    )
    np.testing.assert_array_equal(result["voltages"], np.full(4, -70.0))
    for key in ("theta1", "theta2", "events"):
        np.testing.assert_array_equal(result[key], np.zeros(4))
    assert (result["v_final"], result["theta1_final"], result["theta2_final"]) == (-70.0, 0.0, 0.0)
