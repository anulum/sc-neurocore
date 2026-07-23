# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReservoirConstruction from former test_reservoir.py

"""Focused suite: TestReservoirConstruction from former test_reservoir.py."""

from __future__ import annotations

from tests.reservoir_support import *  # noqa: F403

class TestReservoirConstruction:
    def test_default_construction(self):
        res = AutoCriticalReservoir(n_inputs=2, n_neurons=50, seed=0)
        assert res.n_neurons == 50
        assert res.n_inputs == 2
        assert res.W_res.shape == (50, 50)
        assert res.W_in.shape == (50, 2)
        assert res.W_out.shape == (10, 50)

    def test_no_self_connections(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=20, seed=0)
        np.testing.assert_array_equal(np.diag(res.W_res), 0.0)

    def test_sparsity(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=100, connectivity=0.1, seed=0)
        nonzero_frac = np.count_nonzero(res.W_res) / (100 * 100)
        assert 0.05 < nonzero_frac < 0.15

    def test_critical_weight_formula(self):
        res = AutoCriticalReservoir(
            n_inputs=1,
            n_neurons=100,
            threshold=1.0,
            leak=0.1,
            connectivity=0.1,
            seed=0,
        )
        expected = 1.0 / (2.0 * 0.1 * 100 * 0.1)
        assert abs(res.w_critical - expected) < 1e-10

    def test_spectral_radius_finite(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=30, seed=0)
        sr = res.spectral_radius
        assert np.isfinite(sr)
        assert sr > 0

    def test_deterministic_seed(self):
        a = AutoCriticalReservoir(n_inputs=2, n_neurons=30, seed=99)
        b = AutoCriticalReservoir(n_inputs=2, n_neurons=30, seed=99)
        np.testing.assert_array_equal(a.W_res, b.W_res)
        np.testing.assert_array_equal(a.W_in, b.W_in)
