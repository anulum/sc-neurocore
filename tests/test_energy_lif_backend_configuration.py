# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — configured EnergyLIF polyglot routes

"""Compare configured traces and direct empty-batch refusal across five runtimes."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from sc_neurocore.accel.energy_lif import PARITY_ATOL, backend_available, simulate_energy_lif
from tests.energy_lif_contract_support import (
    CONFIGURATIONS,
    INVALID_CONFIGURATIONS,
    configuration_values,
)
from tests.engine_requirement import require_engine


@pytest.mark.parametrize("parameters", CONFIGURATIONS)
@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_configured_public_backend_trace(parameters: dict[str, float], backend: str) -> None:
    """Exercise every actual runtime with independent and combined field overrides."""
    assert backend_available(backend), f"required {backend} EnergyLIF backend unavailable"
    drive = np.resize(np.array([80.0, 0.0, 120.0, 20.0]), 256)
    reference = simulate_energy_lif(drive, backend="python", **parameters)
    actual = simulate_energy_lif(drive, backend=backend, **parameters)
    np.testing.assert_array_equal(actual["events"], reference["events"])
    for key in ("voltages", "epsilon"):
        values = np.asarray(actual[key])
        assert values.dtype == np.float64 and values.shape == drive.shape
        np.testing.assert_allclose(
            values, np.asarray(reference[key]), atol=PARITY_ATOL[backend], rtol=0
        )
    assert np.asarray(actual["events"]).dtype == np.int64
    assert actual["v_final"] == pytest.approx(reference["v_final"], rel=0, abs=PARITY_ATOL[backend])
    assert actual["epsilon_final"] == pytest.approx(
        reference["epsilon_final"], rel=0, abs=PARITY_ATOL[backend]
    )


@pytest.mark.parametrize("parameters", INVALID_CONFIGURATIONS)
@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_public_dispatch_refuses_invalid_empty_configuration(
    parameters: dict[str, float], backend: str
) -> None:
    """The public dispatcher's Python preflight refuses before selecting any backend."""
    with pytest.raises(ValueError):
        simulate_energy_lif(np.empty(0), backend=backend, **parameters)


@pytest.mark.parametrize("parameters", INVALID_CONFIGURATIONS)
@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("backend", ["rust", "julia", "go", "mojo"])
def test_direct_runtime_refuses_invalid_configuration(
    parameters: dict[str, float], count: int, backend: str
) -> None:
    """Bypass dispatcher preflight through the real lower-level exported batch API."""
    assert backend_available(backend), f"required {backend} EnergyLIF backend unavailable"
    drive = np.full(count, 80.0)
    if backend == "rust":
        extension = require_engine()
        batch = cast(Callable[..., dict[str, object]], extension.py_energy_lif_simulate)
        with pytest.raises(ValueError, match="^invalid EnergyLIF state or configuration$"):
            batch(*configuration_values(parameters), drive)
    elif backend == "julia":
        from sc_neurocore.accel.julia.neurons.energy_lif import simulate_energy_lif as julia

        with pytest.raises(Exception, match="invalid EnergyLIF state or configuration"):
            julia(drive, **parameters)
    else:
        if backend == "go":
            from sc_neurocore.accel.go.energy_lif import simulate_energy_lif as batch_native
        else:
            from sc_neurocore.accel.mojo.energy_lif import simulate_energy_lif as batch_native
        with pytest.raises(FloatingPointError, match="batch failed with status 2$"):
            batch_native(*configuration_values(parameters), drive)
    np.testing.assert_array_equal(drive, np.full(count, 80.0))


@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_valid_empty_trace_and_inclusive_reset_domain(backend: str) -> None:
    """Zero-work batches preserve initial state at inclusive configuration bounds."""
    assert backend_available(backend), f"required {backend} EnergyLIF backend unavailable"
    result = simulate_energy_lif(np.empty(0), backend=backend, e_0=-200.0, alpha=10.0)
    for key in ("voltages", "epsilon", "events"):
        assert np.asarray(result[key]).shape == (0,)
    assert (result["v_final"], result["epsilon_final"]) == (-61.0, 0.32)
