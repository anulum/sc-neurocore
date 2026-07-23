# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSolverBenchmark from former test_solvers_ode.py

"""Focused suite: TestSolverBenchmark from former test_solvers_ode.py."""

from __future__ import annotations

from tests.solvers_ode_support import *  # noqa: F403

class TestSolverBenchmark:
    def test_rk4_throughput(self):
        """100k RK4 steps in < 5s."""
        solver = RK4Solver()
        y = np.array([1.0])
        t0 = time.perf_counter()
        for i in range(100_000):
            y, _ = solver.step(decay_ode, y, 0.0, 1e-5)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"100k RK4 steps took {elapsed:.2f}s"

    def test_exact_lif_throughput(self):
        """1000 exact LIF simulations, 100ms each."""
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)
        t0 = time.perf_counter()
        for _ in range(1000):
            solver.simulate(current=25.0, t_end=100.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"1000 LIF sims took {elapsed:.2f}s"
