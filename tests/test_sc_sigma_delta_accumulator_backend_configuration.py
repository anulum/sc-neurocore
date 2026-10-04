# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured five-runtime bipolar accumulator contracts

"""Exercise complete configured signed accumulator traces on all real backends."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from sc_neurocore.accel.sc_sigma_delta_accumulator import (
    backend_available,
    simulate_sc_sigma_delta_accumulator,
)
from tests.test_sc_sigma_delta_accumulator_engine_binding_configuration import (
    PROFILES,
    _drive,
    _parameters,
)


@pytest.mark.parametrize("overrides", PROFILES)
@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_configured_public_backend_preserves_trace_and_empty_state(
    overrides: dict[str, float],
    backend: str,
) -> None:
    """Retain exact signed events, residuals and empty-input initial state."""
    assert backend_available(backend), (
        f"required configured SC accumulator backend {backend} unavailable"
    )
    parameters = _parameters(overrides)
    expected = simulate_sc_sigma_delta_accumulator(_drive(), **parameters, backend="python")
    actual = simulate_sc_sigma_delta_accumulator(_drive(), **parameters, backend=backend)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    output = np.asarray(actual["sigma"])
    assert output.dtype == np.float64 and output.shape == (256,)
    np.testing.assert_array_equal(output, expected["sigma"])
    assert np.asarray(actual["events"]).dtype == np.int64
    assert actual["sigma_final"] == expected["sigma_final"]
    empty = simulate_sc_sigma_delta_accumulator([], **parameters, backend=backend)
    assert np.asarray(empty["sigma"]).shape == np.asarray(empty["events"]).shape == (0,)
    final = float(cast(float, empty["sigma_final"]))
    assert final == parameters["sigma"]
    assert np.signbit(final) == np.signbit(parameters["sigma"])
