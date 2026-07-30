# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Five-runtime dispatch tests for the retained SC recurrence."""

from __future__ import annotations
import numpy as np
import pytest
from sc_neurocore.accel.sc_non_resetting_adaptive_lif import (
    PARITY_ATOL,
    backend_available,
    simulate_sc_non_resetting_adaptive_lif,
)


def _drive() -> np.ndarray:
    return np.asarray(
        [0.0] * 32 + [20.0] * 96 + [value for _ in range(64) for value in (20.0, 60.0)],
        dtype=np.float64,
    )


@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_available_backend_matches_complete_python_trace(backend: str) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} runtime unavailable")
    expected = simulate_sc_non_resetting_adaptive_lif(_drive(), backend="python")
    actual = simulate_sc_non_resetting_adaptive_lif(_drive(), backend=backend)
    atol = PARITY_ATOL[backend]
    assert np.array_equal(actual["events"], expected["events"])
    for key in ("voltages", "theta"):
        assert np.allclose(actual[key], expected[key], rtol=0.0, atol=atol)


def test_invalid_drive_fails_before_dispatch() -> None:
    with pytest.raises(ValueError, match="finite"):
        simulate_sc_non_resetting_adaptive_lif([0.0, np.inf], backend="python")
