# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExponentialEuler from former test_solvers_ode.py

"""Focused suite: TestExponentialEuler from former test_solvers_ode.py."""

from __future__ import annotations

from numpy.typing import NDArray
import math
import numpy as np
from sc_neurocore.solvers import (
    ExponentialEuler,
)


class TestExponentialEuler:
    """The exponential Euler solver for linear membranes."""

    def test_exact_for_constant_current(self) -> None:
        """ExponentialEuler is exact for linear LIF with constant I."""
        tau = 20.0
        v_rest = -65.0
        solver = ExponentialEuler(tau=tau, y_rest=v_rest, r_m=1.0)

        def current_fn(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([10.0])

        y = np.array([v_rest])
        dt = 5.0
        y_new, _ = solver.step(current_fn, y, 0.0, dt)
        expected = v_rest + 10.0 * (1.0 - math.exp(-dt / tau))
        assert abs(y_new[0] - expected) < 1e-10
