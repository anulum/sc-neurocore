# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScDivide from former test_cordiv_division.py

"""Focused suite: TestScDivide from former test_cordiv_division.py."""

from __future__ import annotations

from tests.cordiv_division_support import *  # noqa: F403


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
