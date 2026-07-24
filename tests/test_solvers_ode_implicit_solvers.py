# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImplicitSolvers from former test_solvers_ode.py

"""Focused suite: TestImplicitSolvers from former test_solvers_ode.py."""

from __future__ import annotations

from tests.solvers_ode_support import *  # noqa: F403


class TestImplicitSolvers:
    def test_rosenbrock_euler_stable_for_stiff(self):
        solver = RosenbrockEuler()
        y = np.array([1.0])
        dt = 0.01
        for _ in range(10):
            y, _ = solver.step(stiff_ode, y, 0.0, dt)
        assert 0.0 <= y[0] < 1e-8

    def test_rosenbrock_euler_rejects_invalid_parameters(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            RosenbrockEuler(gamma=0.0)
        with pytest.raises(ValueError, match="jacobian_epsilon must be positive"):
            RosenbrockEuler(jacobian_epsilon=0.0)

    def test_rosenbrock_euler_falls_back_to_lstsq_on_singular_system(self, monkeypatch):
        # When the Newton matrix (I - gamma*dt*J) is singular, np.linalg.solve
        # raises LinAlgError; the step must recover via a least-squares solve.
        # A finite-difference Jacobian rarely lands exactly singular, so force
        # the failure deterministically and assert the fallback still returns a
        # finite increment.
        import sc_neurocore.solvers.stiff as stiff_mod

        def _raise_singular(*_args, **_kwargs):
            raise np.linalg.LinAlgError("forced singular system")

        monkeypatch.setattr(stiff_mod.np.linalg, "solve", _raise_singular)
        solver = RosenbrockEuler(gamma=1.0)
        y_new, dt_used = solver.step(lambda t, y: y, np.array([1.0]), 0.0, 0.5)
        assert np.all(np.isfinite(y_new))
        assert dt_used == 0.5

    def test_implicit_euler_stable_for_stiff(self):
        solver = ImplicitEuler(max_iterations=50)
        y = np.array([1.0])
        dt = 0.001
        for _ in range(1000):
            y, _ = solver.step(stiff_ode, y, 0.0, dt)
        assert abs(y[0]) < 1e-3  # decayed

    def test_trapezoidal_stable_for_stiff(self):
        solver = TrapezoidalRule(max_iterations=50)
        y = np.array([1.0])
        dt = 0.001
        for _ in range(1000):
            y, _ = solver.step(stiff_ode, y, 0.0, dt)
        assert abs(y[0]) < 1e-3

    def test_trapezoidal_more_accurate_than_implicit_euler(self):
        y0 = np.array([1.0])
        dt = 0.01
        exact = math.exp(-10.0)  # t=10*dt=0.1 for standard decay
        n = 10

        ye = y0.copy()
        ie = ImplicitEuler()
        for _ in range(n):
            ye, _ = ie.step(decay_ode, ye, 0.0, dt)

        yt = y0.copy()
        tr = TrapezoidalRule()
        for _ in range(n):
            yt, _ = tr.step(decay_ode, yt, 0.0, dt)

        assert abs(yt[0] - exact) < abs(ye[0] - exact)
