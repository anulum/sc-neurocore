# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChaoticRNG from former test_chaos.py

"""Focused suite: TestChaoticRNG from former test_chaos.py."""

from __future__ import annotations

from tests.chaos_support import *  # noqa: F403

class TestChaoticRNG:
    def test_output_range(self):
        rng = ChaoticRNG(r=4.0, x=0.37)
        vals = rng.random(10_000)
        assert vals.min() > 0.0
        assert vals.max() < 1.0

    def test_deterministic(self):
        a = ChaoticRNG(r=4.0, x=0.37)
        b = ChaoticRNG(r=4.0, x=0.37)
        np.testing.assert_array_equal(a.random(100), b.random(100))

    def test_different_seeds_diverge(self):
        a = ChaoticRNG(r=4.0, x=0.37)
        b = ChaoticRNG(r=4.0, x=0.38)
        assert not np.allclose(a.random(100), b.random(100))

    def test_bitstream_probability(self):
        rng = ChaoticRNG(r=4.0, x=0.37)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            bits = rng.generate_bitstream(p, 50_000)
            assert bits.dtype == np.uint8
            assert set(np.unique(bits)).issubset({0, 1})
            # After Beta→uniform CDF correction, tolerance ±3%
            assert abs(bits.mean() - p) < 0.03, f"p={p}, got {bits.mean():.3f}"

    def test_lyapunov_exponent_at_r4(self):
        rng = ChaoticRNG(r=4.0, x=0.37)
        lam = rng.lyapunov_exponent(n_steps=50_000)
        # Theoretical: ln(2) ~ 0.6931
        assert abs(lam - np.log(2)) < 0.02, f"Lyapunov={lam:.4f}, expected ~0.6931"

    def test_lyapunov_positive_in_chaos(self):
        rng = ChaoticRNG(r=3.9, x=0.37)
        lam = rng.lyapunov_exponent(n_steps=10_000)
        assert lam > 0.0

    def test_shannon_entropy(self):
        rng = ChaoticRNG(r=4.0, x=0.37)
        entropy = rng.shannon_entropy(n_samples=50_000, n_bins=100)
        # For 100 bins, uniform entropy = log2(100) ~ 6.64
        # Beta(0.5,0.5) is slightly below uniform
        assert entropy > 6.0
        assert entropy < 7.0

    def test_autocorrelation_near_zero(self):
        rng = ChaoticRNG(r=4.0, x=0.37)
        acf = rng.autocorrelation(n_samples=50_000, max_lag=20)
        assert acf[0] == pytest.approx(1.0)
        # All lags > 0 should be near zero
        assert np.abs(acf[1:]).max() < 0.05

    def test_reset_reproduces(self):
        rng = ChaoticRNG(r=4.0, x=0.37)
        first = rng.random(50)
        rng.reset()
        second = rng.random(50)
        np.testing.assert_array_equal(first, second)

    def test_vectorized_same_length(self):
        rng = ChaoticRNG(r=4.0, x=0.37)
        vals = rng.random_vectorized(1000, n_maps=4)
        assert vals.shape == (1000,)
        assert vals.min() > 0.0
        assert vals.max() < 1.0

    def test_vectorized_distribution(self):
        rng = ChaoticRNG(r=4.0, x=0.37)
        vals = rng.random_vectorized(50_000, n_maps=8)
        assert abs(vals.mean() - 0.5) < 0.05

    def test_invalid_x(self):
        with pytest.raises(ValueError, match="x must be in"):
            ChaoticRNG(r=4.0, x=0.0)
        with pytest.raises(ValueError, match="x must be in"):
            ChaoticRNG(r=4.0, x=1.0)

    def test_invalid_r(self):
        with pytest.raises(ValueError, match="r must be in"):
            ChaoticRNG(r=2.0, x=0.5)

    def test_state_property(self):
        rng = ChaoticRNG(r=4.0, x=0.37)
        s = rng.state
        assert 0.0 < s < 1.0
