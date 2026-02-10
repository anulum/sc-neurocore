"""Tests for Rust-accelerated Kuramoto solver."""

import numpy as np

from sc_neurocore_engine import KuramotoSolver


class TestKuramotoSolver:
    def test_synchronization(self):
        n = 100
        omega = np.ones(n)
        coupling = np.full((n, n), 1.0)
        phases = np.random.RandomState(42).uniform(0, 2 * np.pi, n)

        solver = KuramotoSolver(omega, coupling, phases, noise_amp=0.0)
        order_values = solver.run(n_steps=500, dt=0.01)

        assert order_values[-1] > 0.8, (
            f"Strong coupling should synchronize: R={order_values[-1]:.4f}"
        )

    def test_order_parameter_range(self):
        solver = KuramotoSolver(
            np.ones(50),
            np.zeros((50, 50)),
            np.random.RandomState(42).uniform(0, 2 * np.pi, 50),
        )
        order_value = solver.order_parameter()
        assert 0.0 <= order_value <= 1.0

    def test_phase_roundtrip(self):
        phases = np.array([0.1, 0.2, 0.3, 0.4])
        solver = KuramotoSolver(np.ones(4), np.zeros((4, 4)), phases)
        np.testing.assert_allclose(solver.phases, phases, atol=1e-12)
