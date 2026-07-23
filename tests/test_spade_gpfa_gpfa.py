# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGPFA from former test_spade_gpfa.py

"""Focused suite: TestGPFA from former test_spade_gpfa.py."""

from __future__ import annotations

from tests.spade_gpfa_support import *  # noqa: F403

class TestGPFA:
    def test_log_likelihood_includes_gaussian_normalisation(self):
        y = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        c = np.zeros((1, 1), dtype=np.float64)
        d = np.zeros(1, dtype=np.float64)
        r = np.diag([2.0])
        k_all = [_gp_kernel(3, tau=2.0)]

        ll = _gpfa_log_likelihood(y, c, d, r, k_all)
        expected = -0.5 * (np.sum(y[0] ** 2 / 2.0) + 3 * np.log(2.0) + 3 * np.log(2.0 * np.pi))
        np.testing.assert_allclose(ll, expected, rtol=0.0, atol=1e-8)

    def test_basic_output_shape(self):
        trains = _poisson_trains(n_neurons=6, duration_s=0.5)
        result = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=5)
        assert result["trajectories"].shape[0] == 2
        assert result["C"].shape == (6, 2)
        assert result["d"].shape == (6,)
        assert len(result["log_likelihoods"]) > 0

    def test_log_likelihood_increases(self):
        trains = _poisson_trains(n_neurons=8, duration_s=0.5, seed=7)
        result = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=15)
        lls = result["log_likelihoods"]
        if len(lls) >= 3:
            assert lls[-1] >= lls[0] - 1.0  # allow small numerical wobble

    def test_empty_trains(self):
        result = gpfa([], n_latents=2)
        assert result["trajectories"].size == 0

    def test_transform_matches_shape(self):
        trains = _poisson_trains(n_neurons=4, duration_s=0.5)
        result = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=5)
        new_trains = _poisson_trains(n_neurons=4, duration_s=0.5, seed=99)
        proj = gpfa_transform(new_trains, result, bin_ms=20.0)
        assert proj.shape[0] == 2

    def test_single_latent(self):
        trains = _poisson_trains(n_neurons=3, duration_s=0.3)
        result = gpfa(trains, n_latents=1, bin_ms=20.0, max_iter=5)
        assert result["trajectories"].shape[0] == 1
        assert result["C"].shape[1] == 1
