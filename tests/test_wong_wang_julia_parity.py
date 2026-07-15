# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Julia Wong-Wang Euler/OU parity

"""Compare the Julia batch with the deterministic Python golden."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.julia.neurons import (
    _HAS_JULIA_NEURONS,
    _ensure_wong_wang_loaded,
    simulate_wong_wang,
)
from sc_neurocore.accel.wong_wang import simulate_python

_PARAMETERS = (0.24, 0.11, 0.01, -0.02, 0.12, 0.003, 0.7, 0.28, 0.06, 0.31, 0.015, 0.0002)


def _inputs(steps: int) -> tuple[npt.NDArray[np.float64], ...]:
    index = np.arange(steps, dtype=np.float64)
    return (
        0.02 + 0.01 * np.sin(index * 0.07),
        -0.01 + 0.008 * np.cos(index * 0.11),
        np.sin(np.arange(2 * steps, dtype=np.float64) * 0.17),
    )


def test_julia_runtime_and_kernel_are_available() -> None:
    """Keep the maintained Julia lane fail-closed."""
    assert _HAS_JULIA_NEURONS
    assert _ensure_wong_wang_loaded() is not None


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_julia_complete_mapping_matches_python(steps: int) -> None:
    """Compare all traces and final receipts under one explicit sample stream."""
    inputs = _inputs(steps)
    expected = simulate_python(*_PARAMETERS, *inputs)
    actual = simulate_wong_wang(*_PARAMETERS, *inputs)
    for key in ("s1", "s2", "noise1", "noise2", "r1", "r2"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
    for key in ("s1_final", "s2_final", "noise1_final", "noise2_final"):
        assert float(actual[key]) == pytest.approx(float(expected[key]), abs=1.0e-12)


def test_julia_rejects_non_finite_samples_before_dispatch() -> None:
    """Keep array validation on the Python side of the runtime bridge."""
    stim1, stim2, xi = _inputs(2)
    stim2[1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        simulate_wong_wang(*_PARAMETERS, stim1, stim2, xi)


def test_julia_rejects_length_mismatch_before_dispatch() -> None:
    """Reject incomplete interleaved sample vectors deterministically."""
    with pytest.raises(ValueError, match="xi length"):
        simulate_wong_wang(*_PARAMETERS, np.zeros(2), np.zeros(2), np.zeros(2))
