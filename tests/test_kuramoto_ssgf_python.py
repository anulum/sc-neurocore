"""Tests for SSGF-compatible Kuramoto solver (Python bridge)."""

import numpy as np

from sc_neurocore_engine import KuramotoSolver


class TestSSGFKuramoto:
    def test_step_ssgf_matches_step_without_extras(self):
        n = 16
        omega = np.ones(n)
        coupling = np.full((n, n), 0.3)
        phases = np.arange(n) * 0.3

        solver_a = KuramotoSolver(omega, coupling, phases.copy(), noise_amp=0.0)
        solver_b = KuramotoSolver(omega, coupling, phases.copy(), noise_amp=0.0)

        r_basic = solver_a.step(0.01, seed=42)
        r_ssgf = solver_b.step_ssgf(0.01, seed=42)

        assert abs(r_basic - r_ssgf) < 1e-14
        np.testing.assert_allclose(solver_a.phases, solver_b.phases, atol=1e-14)

    def test_geometry_coupling_changes_output(self):
        n = 20
        omega = np.ones(n)
        coupling = np.full((n, n), 0.5)
        phases = np.random.RandomState(42).uniform(0, 2 * np.pi, n)

        solver = KuramotoSolver(omega, coupling, phases.copy(), noise_amp=0.0)
        W = np.ones((n, n))

        r_values = solver.run_ssgf(
            n_steps=100,
            dt=0.01,
            W=W,
            sigma_g=0.5,
        )
        assert len(r_values) == 100
        assert r_values[-1] > r_values[0]

    def test_field_pressure(self):
        n = 20
        omega = np.zeros(n)
        coupling = np.zeros((n, n))
        phases = np.random.RandomState(42).uniform(0, 2 * np.pi, n)

        solver = KuramotoSolver(omega, coupling, phases.copy(), noise_amp=0.0)
        solver.set_field_pressure(1.0)

        solver.run_ssgf(n_steps=500, dt=0.01)
        r = solver.order_parameter()
        assert r > 0.1, f"Field pressure should create structure, R={r}"

    def test_pgbo_coupling(self):
        n = 20
        omega = np.ones(n)
        coupling = np.full((n, n), 0.3)
        phases = np.random.RandomState(42).uniform(0, 2 * np.pi, n)

        solver = KuramotoSolver(omega, coupling, phases.copy(), noise_amp=0.0)
        h = np.eye(n) * 2.0

        r = solver.step_ssgf(
            0.01,
            W=None,
            sigma_g=0.0,
            h_munu=h,
            pgbo_weight=1.0,
        )
        assert 0.0 <= r <= 1.0
