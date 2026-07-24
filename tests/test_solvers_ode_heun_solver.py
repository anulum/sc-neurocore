# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHeunSolver from former test_solvers_ode.py

"""Focused suite: TestHeunSolver from former test_solvers_ode.py."""

from __future__ import annotations

from tests.solvers_ode_support import *  # noqa: F403


class TestHeunSolver:
    def test_convergence_order_2(self):
        y0 = np.array([1.0])
        t_end = 1.0
        exact = math.exp(-t_end)
        errors = []
        for n_steps in [50, 100, 200]:
            dt = t_end / n_steps
            y = y0.copy()
            solver = HeunSolver()
            for i in range(n_steps):
                y, _ = solver.step(decay_ode, y, i * dt, dt)
            errors.append(abs(y[0] - exact))
        ratio = errors[0] / errors[1]
        assert 3.0 < ratio < 5.0  # O(h²) → ratio ~4
