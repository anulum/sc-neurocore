# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Go Wong-Wang Euler/OU parity

"""Compare the Go C-shared batch with the deterministic Python golden."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.go.wong_wang import _HAS_GO_WONG_WANG, simulate_wong_wang
from sc_neurocore.accel.wong_wang import simulate_python

_PARAMETERS = (0.24, 0.11, 0.01, -0.02, 0.12, 0.003, 0.7, 0.28, 0.06, 0.31, 0.015, 0.0002)


def _inputs(steps: int) -> tuple[npt.NDArray[np.float64], ...]:
    index = np.arange(steps, dtype=np.float64)
    return (
        0.02 + 0.01 * np.sin(index * 0.07),
        -0.01 + 0.008 * np.cos(index * 0.11),
        np.sin(np.arange(2 * steps, dtype=np.float64) * 0.17),
    )


def test_go_shared_library_is_available() -> None:
    """Keep the maintained native lane fail-closed."""
    assert _HAS_GO_WONG_WANG


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_go_complete_mapping_matches_python(steps: int) -> None:
    """Compare all traces and final receipts under one explicit sample stream."""
    inputs = _inputs(steps)
    expected = simulate_python(*_PARAMETERS, *inputs)
    actual = simulate_wong_wang(*_PARAMETERS, *inputs)
    for key in ("s1", "s2", "noise1", "noise2", "r1", "r2"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
    for key in ("s1_final", "s2_final", "noise1_final", "noise2_final"):
        assert float(actual[key]) == pytest.approx(float(expected[key]), abs=1.0e-12)


def test_go_rejects_non_finite_samples() -> None:
    """Reject invalid arrays before the C boundary."""
    stim1, stim2, xi = _inputs(2)
    xi[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        simulate_wong_wang(*_PARAMETERS, stim1, stim2, xi)


def test_go_native_error_is_reported() -> None:
    """Map a rejected scalar configuration into a Python runtime error."""
    invalid = (*_PARAMETERS[:-1], -0.0002)
    with pytest.raises(RuntimeError, match="code 2"):
        simulate_wong_wang(*invalid, *_inputs(1))
