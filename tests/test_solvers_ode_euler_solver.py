# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEulerSolver from former test_solvers_ode.py

"""Focused suite: TestEulerSolver from former test_solvers_ode.py."""

from __future__ import annotations

from tests.solvers_ode_support import *  # noqa: F403


class TestEulerSolver:
    def test_decay_direction(self):
        solver = EulerSolver()
        y = np.array([1.0])
        y_new, _ = solver.step(decay_ode, y, t=0.0, dt=0.01)
        assert y_new[0] < 1.0

    def test_convergence_order_1(self):
        """Euler error should be O(h)."""
        y0 = np.array([1.0])
        t_end = 1.0
        exact = math.exp(-t_end)
        errors = []
        for n_steps in [100, 200, 400]:
            dt = t_end / n_steps
            y = y0.copy()
            solver = EulerSolver()
            for _ in range(n_steps):
                y, _ = solver.step(decay_ode, y, 0.0, dt)
            errors.append(abs(y[0] - exact))
        # Error ratio should be ~2 for O(h)
        ratio = errors[0] / errors[1]
        assert 1.5 < ratio < 2.5
