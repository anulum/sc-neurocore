# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDormandPrinceSolver from former test_solvers_ode.py

"""Focused suite: TestDormandPrinceSolver from former test_solvers_ode.py."""

from __future__ import annotations

from numpy.typing import NDArray
import math
import numpy as np
import pytest
from sc_neurocore.solvers import (
    DormandPrinceSolver,
)
from tests.solvers_ode_support import (
    decay_ode,
)


class TestDormandPrinceSolver:
    """The adaptive Dormand-Prince solver."""

    def test_adaptive_reaches_solution(self) -> None:
        """Adaptive steps reach the exact decay value."""
        solver = DormandPrinceSolver(atol=1e-8, rtol=1e-6)
        ts, ys = solver.integrate(decay_ode, np.array([1.0]), (0.0, 1.0))
        assert abs(ys[-1, 0] - math.exp(-1.0)) < 1e-5

    def test_step_size_adapts(self) -> None:
        """The step size varies over the run."""
        solver = DormandPrinceSolver()
        ts, ys = solver.integrate(decay_ode, np.array([1.0]), (0.0, 2.0), dt0=0.001)
        dts = np.diff(ts)
        assert dts.max() > dts.min() * 1.5  # step varies

    def test_high_precision(self) -> None:
        """Tight tolerances reach high precision."""
        solver = DormandPrinceSolver(atol=1e-12, rtol=1e-10)
        ts, ys = solver.integrate(decay_ode, np.array([1.0]), (0.0, 1.0))
        assert abs(ys[-1, 0] - math.exp(-1.0)) < 1e-9

    def test_zero_error_uses_max_growth_factor(self) -> None:
        """A zero error estimate grows the step by the maximum factor."""
        solver = DormandPrinceSolver(max_factor=3.0)

        def zero_rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.zeros_like(y)

        y_new, dt_used, dt_next = solver.step(zero_rhs, np.array([1.0]), 0.0, 0.1)

        np.testing.assert_allclose(y_new, np.array([1.0]))
        assert dt_used == pytest.approx(0.1)
        assert dt_next == pytest.approx(0.3)

    def test_rejects_initial_step_when_error_is_large(self) -> None:
        """A large error shrinks the first step."""
        solver = DormandPrinceSolver(atol=1e-12, rtol=1e-12, min_factor=0.2)

        y_new, dt_used, dt_next = solver.step(lambda t, y: y * y, np.array([1.0]), 0.0, 1.0)

        assert dt_used < 1.0
        assert dt_next > 0.0
        assert y_new[0] > 1.0
