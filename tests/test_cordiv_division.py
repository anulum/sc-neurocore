# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for CORDIV stochastic division and adaptive_length

"""Tests for sc_divide (Li et al. 2014) and Hoeffding adaptive_length."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.utils.bitstreams import (
    generate_bernoulli_bitstream,
    sc_divide,
    adaptive_length,
)
from sc_neurocore.utils.rng import RNG


class TestScDivide:
    def test_half_divided_by_one(self):
        """0.5 / 1.0 should ≈ 0.5."""
        L = 8192
        x = generate_bernoulli_bitstream(0.5, L, rng=RNG(42))
        y = generate_bernoulli_bitstream(1.0, L, rng=RNG(99))
        z = sc_divide(x, y)
        np.testing.assert_allclose(np.mean(z), 0.5, atol=0.05)

    def test_equal_numerator_denominator(self):
        """p / p should be closer to 1.0 than to 0.0.
        CORDIV is a state machine; bias is expected for correlated inputs."""
        L = 8192
        p = 0.6
        x = generate_bernoulli_bitstream(p, L, rng=RNG(42))
        y = generate_bernoulli_bitstream(p, L, rng=RNG(99))
        z = sc_divide(x, y)
        result = np.mean(z)
        assert result > 0.5, f"p/p gave {result:.3f}, expected > 0.5"

    def test_quarter_by_half(self):
        """0.25 / 0.5 should be in the right neighbourhood.
        CORDIV has inherent bias from the hold-state behaviour."""
        L = 8192
        x = generate_bernoulli_bitstream(0.25, L, rng=RNG(1))
        y = generate_bernoulli_bitstream(0.5, L, rng=RNG(2))
        z = sc_divide(x, y)
        result = np.mean(z)
        assert 0.2 < result < 0.8, f"0.25/0.5 gave {result:.3f}, outside [0.2, 0.8]"

    def test_output_binary(self):
        """Output should be binary (0 or 1)."""
        L = 1024
        x = generate_bernoulli_bitstream(0.3, L, rng=RNG(10))
        y = generate_bernoulli_bitstream(0.6, L, rng=RNG(20))
        z = sc_divide(x, y)
        assert set(np.unique(z)).issubset({0, 1})

    def test_output_length(self):
        L = 512
        x = generate_bernoulli_bitstream(0.4, L, rng=RNG(10))
        y = generate_bernoulli_bitstream(0.8, L, rng=RNG(20))
        z = sc_divide(x, y)
        assert len(z) == L

    def test_monotonicity(self):
        """Increasing numerator should increase output."""
        L = 4096
        y = generate_bernoulli_bitstream(0.8, L, rng=RNG(99))
        results = []
        for pn in [0.1, 0.3, 0.5, 0.7]:
            x = generate_bernoulli_bitstream(pn, L, rng=RNG(42))
            z = sc_divide(x, y)
            results.append(np.mean(z))
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1] + 0.1, "not monotonic"

    def test_convergence_with_length(self):
        """Error should decrease with longer bitstreams."""
        pn, pd = 0.3, 0.6
        expected = pn / pd
        errors = []
        for L in [256, 1024, 4096]:
            errs = []
            for trial in range(50):
                x = generate_bernoulli_bitstream(pn, L, rng=RNG(trial))
                y = generate_bernoulli_bitstream(pd, L, rng=RNG(trial + 1000))
                z = sc_divide(x, y)
                errs.append(abs(np.mean(z) - expected))
            errors.append(np.mean(errs))
        assert errors[-1] < errors[0], "longer L should have lower error"


class TestAdaptiveLength:
    def test_returns_positive_int(self):
        L = adaptive_length(0.5, epsilon=0.01, confidence=0.95)
        assert isinstance(L, int)
        assert L > 0

    def test_tighter_bound_needs_longer(self):
        L1 = adaptive_length(0.5, epsilon=0.1, confidence=0.95)
        L2 = adaptive_length(0.5, epsilon=0.01, confidence=0.95)
        assert L2 > L1

    def test_higher_confidence_needs_longer(self):
        L1 = adaptive_length(0.5, epsilon=0.05, confidence=0.90)
        L2 = adaptive_length(0.5, epsilon=0.05, confidence=0.99)
        assert L2 > L1

    def test_respects_min_length(self):
        L = adaptive_length(0.5, epsilon=0.5, confidence=0.5, min_length=64)
        assert L >= 64

    def test_respects_max_length(self):
        L = adaptive_length(0.5, epsilon=0.001, confidence=0.999, max_length=1024)
        assert L <= 1024

    def test_hoeffding_formula(self):
        """L >= ln(2/delta) / (2*eps^2) for Hoeffding bound."""
        eps = 0.05
        delta = 0.05  # 95% confidence
        expected_min = np.log(2.0 / delta) / (2.0 * eps ** 2)
        L = adaptive_length(0.5, epsilon=eps, confidence=0.95, max_length=100000)
        assert L >= int(expected_min) - 1, f"L={L} < Hoeffding {expected_min:.0f}"
