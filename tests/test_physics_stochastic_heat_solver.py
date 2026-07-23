# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticHeatSolver from former test_physics.py

"""Focused suite: TestStochasticHeatSolver from former test_physics.py."""

from __future__ import annotations

from tests.physics_support import *  # noqa: F403

class TestStochasticHeatSolver:
    def test_construction(self):
        s = _make_uniform_solver(length=100, num_walkers=500, diffusivity=0.1)
        assert len(s.walkers) == 500
        assert s.length == 100

    def test_step_moves_walkers(self):
        s = _make_uniform_solver(length=100, num_walkers=100, diffusivity=0.1, seed=42)
        pos_before = s.walkers.copy()
        s.step()
        assert not np.array_equal(s.walkers, pos_before)

    def test_walkers_stay_in_bounds(self):
        s = _make_uniform_solver(length=50, num_walkers=1000, diffusivity=0.1, seed=0)
        for _ in range(100):
            s.step()
        assert s.walkers.min() >= 0
        assert s.walkers.max() <= 50

    def test_temperature_profile_shape(self):
        s = _make_uniform_solver(length=20, num_walkers=500, diffusivity=0.1)
        profile = s.get_density(n_bins=20)
        assert profile.shape == (20,)

    def test_temperature_sums_to_one(self):
        s = _make_uniform_solver(length=20, num_walkers=1000, diffusivity=0.1)
        profile = s.get_density(n_bins=20)
        bin_width = s.length / 20
        assert abs(profile.sum() * bin_width - 1.0) < 0.01

    def test_diffusion_spreads(self):
        s = StochasticHeatSolver(length=100, num_walkers=10000, diffusivity=0.1, dt=1e-2, seed=42)
        s.set_initial_delta(50.0)
        p0 = s.get_density(n_bins=100)
        for _ in range(200):
            s.step()
        p1 = s.get_density(n_bins=100)
        # Initial delta concentrates density at one bin; diffusion spreads it.
        assert p1.max() < p0.max()
