# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRK4Solver from former test_solvers_ode.py

"""Focused suite: TestRK4Solver from former test_solvers_ode.py."""

from __future__ import annotations

from tests.solvers_ode_support import *  # noqa: F403

class TestRK4Solver:
    def test_convergence_order_4(self):
        y0 = np.array([1.0])
        t_end = 1.0
        exact = math.exp(-t_end)
        errors = []
        for n_steps in [10, 20, 40]:
            dt = t_end / n_steps
            y = y0.copy()
            solver = RK4Solver()
            for i in range(n_steps):
                y, _ = solver.step(decay_ode, y, i * dt, dt)
            errors.append(abs(y[0] - exact))
        ratio = errors[0] / errors[1]
        assert 12.0 < ratio < 20.0  # O(h⁴) → ratio ~16

    def test_accuracy_better_than_euler(self):
        y0 = np.array([1.0])
        n_steps = 20
        dt = 1.0 / n_steps
        exact = math.exp(-1.0)

        y_e = y0.copy()
        euler = EulerSolver()
        for _ in range(n_steps):
            y_e, _ = euler.step(decay_ode, y_e, 0.0, dt)

        y_r = y0.copy()
        rk4 = RK4Solver()
        for i in range(n_steps):
            y_r, _ = rk4.step(decay_ode, y_r, i * dt, dt)

        assert abs(y_r[0] - exact) < abs(y_e[0] - exact)
