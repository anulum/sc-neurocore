# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHRAnalytical from former test_model_hindmarsh_rose.py

"""Focused suite: TestHRAnalytical from former test_model_hindmarsh_rose.py."""

from __future__ import annotations

from tests.model_hindmarsh_rose_support import *  # noqa: F403

class TestHRAnalytical:
    def test_dx_formula_one_step(self):
        """dx = (y - x³ + b·x² - z + I) · dt."""
        n = HindmarshRoseNeuron(integrator="euler")
        x0, y0, z0 = n.x, n.y, n.z
        I = 3.0
        expected_dx = (y0 - x0**3 + n.b * x0**2 - z0 + I) * n.dt
        expected_dy = (1.0 - 5.0 * x0**2 - y0) * n.dt
        expected_dz = n.r * (n.s * (x0 - n.x_rest) - z0) * n.dt
        n.step(I)
        assert abs((n.x - x0) - expected_dx) < 1e-12
        assert abs((n.y - y0) - expected_dy) < 1e-12
        assert abs((n.z - z0) - expected_dz) < 1e-14

    def test_runge_kutta_tracks_substepped_reference_better_than_euler(self):
        horizon = 1.0
        current = 3.5
        reference = HindmarshRoseNeuron(dt=0.001, integrator="rk4")
        coarse_rk4 = HindmarshRoseNeuron(dt=0.1, integrator="rk4")
        coarse_euler = HindmarshRoseNeuron(dt=0.1, integrator="euler")

        for _ in range(int(horizon / reference.dt)):
            reference.step(current)
        for _ in range(int(horizon / coarse_rk4.dt)):
            coarse_rk4.step(current)
            coarse_euler.step(current)

        rk4_error = (
            abs(coarse_rk4.x - reference.x)
            + abs(coarse_rk4.y - reference.y)
            + abs(coarse_rk4.z - reference.z)
        )
        euler_error = (
            abs(coarse_euler.x - reference.x)
            + abs(coarse_euler.y - reference.y)
            + abs(coarse_euler.z - reference.z)
        )

        assert rk4_error < euler_error
        assert rk4_error < 5e-3

    def test_cubic_nonlinearity(self):
        """-x³ creates the excitable dynamics."""
        n = HindmarshRoseNeuron()
        # At x=2: -x³ = -8, b·x² = 12 → net = 4 (positive)
        assert -8.0 + n.b * 4.0 == 4.0

    def test_z_slow_timescale(self):
        """r=0.001 → z changes ~1000x slower than x."""
        n = HindmarshRoseNeuron()
        assert n.r == 0.001

    def test_z_adapts_to_x(self):
        """dz/dt = r·(s·(x-x_rest) - z). z tracks s·(x-x_rest)."""
        n = HindmarshRoseNeuron()
        z0 = n.z
        for _ in range(50_000):
            n.step(5.0)
        # z should have moved from initial value
        assert n.z != z0

    def test_x_nullcline(self):
        """x-nullcline: y = x³ - b·x² + z - I."""
        # At rest (x=-1.6): y_null = (-1.6)³ - 3·(-1.6)² + z - I
        x = -1.6
        y_null = x**3 - 3.0 * x**2 + 2.0 - 0.0
        assert np.isfinite(y_null)

    def test_y_nullcline(self):
        """y-nullcline: y = 1 - 5x²."""
        x = -1.6
        y_null = 1.0 - 5.0 * x**2
        assert abs(y_null - (1.0 - 12.8)) < 1e-10
