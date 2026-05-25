# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Rust-accelerated Kuramoto solver

"""Tests for Rust-accelerated Kuramoto solver."""

import numpy as np

import pytest

pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)

from sc_neurocore_engine import KuramotoSolver


class TestKuramotoSolver:
    def test_synchronization(self):
        n = 100
        omega = np.ones(n)
        coupling = np.full((n, n), 1.0)
        phases = np.random.RandomState(42).uniform(0, 2 * np.pi, n)

        solver = KuramotoSolver(omega, coupling, phases, noise_amp=0.0)
        order_values = solver.run(n_steps=2000, dt=0.01)

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
        updated = phases + 0.5
        solver.phases = updated
        np.testing.assert_allclose(solver.phases, updated, atol=1e-12)


def test_python_bridge_rejects_non_finite_inputs():
    with pytest.raises(ValueError, match="omega values must be finite"):
        KuramotoSolver([1.0, np.nan], np.zeros((2, 2)), [0.1, 0.2], noise_amp=0.0)

    with pytest.raises(ValueError, match="coupling values must be finite"):
        KuramotoSolver([1.0, 1.1], [[0.0, np.inf], [0.0, 0.0]], [0.1, 0.2], noise_amp=0.0)

    with pytest.raises(ValueError, match="initial_phases values must be finite"):
        KuramotoSolver([1.0, 1.1], np.zeros((2, 2)), [0.1, np.nan], noise_amp=0.0)

    with pytest.raises(ValueError, match="noise_amp must be finite and non-negative"):
        KuramotoSolver([1.0, 1.1], np.zeros((2, 2)), [0.1, 0.2], noise_amp=-0.1)


def test_python_bridge_rejects_invalid_runtime_values():
    solver = KuramotoSolver([1.0, 1.1], np.zeros((2, 2)), [0.1, 0.2], noise_amp=0.0)

    with pytest.raises(ValueError, match="dt must be finite and positive"):
        solver.step(0.0)

    with pytest.raises(ValueError, match="phases values must be finite"):
        solver.phases = [0.1, np.nan]

    with pytest.raises(ValueError, match="field_pressure must be finite"):
        solver.set_field_pressure(np.nan)

    with pytest.raises(ValueError, match="w_flat values must be finite"):
        solver.step_ssgf(0.01, W=[[0.0, np.nan], [0.0, 0.0]], sigma_g=1.0)
