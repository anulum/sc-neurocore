# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFactory from former test_solvers_ode.py

"""Focused suite: TestFactory from former test_solvers_ode.py."""

from __future__ import annotations

from tests.solvers_ode_support import *  # noqa: F403


class TestFactory:
    @pytest.mark.parametrize("name", ["euler", "heun", "rk4"])
    def test_get_solver(self, name):
        solver = get_solver(name)
        y, _ = solver.step(decay_ode, np.array([1.0]), 0.0, 0.01)
        assert y[0] < 1.0

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError):
            get_solver("nonexistent_solver")

    def test_dp45_with_kwargs(self):
        solver = get_solver("dp45", atol=1e-6, rtol=1e-4)
        assert isinstance(solver, DormandPrinceSolver)

    def test_exponential_euler_with_kwargs(self):
        solver = get_solver("exponential_euler", tau=5.0, y_rest=-60.0)
        assert isinstance(solver, ExponentialEuler)
        assert solver.tau == pytest.approx(5.0)
        assert solver.y_rest == pytest.approx(-60.0)

    @pytest.mark.parametrize("name", ["rosenbrock", "rosenbrock_euler"])
    def test_rosenbrock_aliases(self, name):
        solver = get_solver(name, gamma=0.5)
        assert isinstance(solver, RosenbrockEuler)
        assert solver.gamma == pytest.approx(0.5)
