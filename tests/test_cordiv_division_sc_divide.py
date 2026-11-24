# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScDivide from former test_cordiv_division.py

"""Focused suite: TestScDivide from former test_cordiv_division.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.utils.bitstreams import (
    generate_bernoulli_bitstream,
    sc_divide,
)
from sc_neurocore.utils.rng import (
    RNG,
)


class TestScDivide:
    """``sc_divide`` against its contract on correlated and independent streams."""

    def test_half_divided_by_one(self) -> None:
        """0.5 / 1.0 should ≈ 0.5."""
        L = 8192
        x = generate_bernoulli_bitstream(0.5, L, rng=RNG(42))
        y = generate_bernoulli_bitstream(1.0, L, rng=RNG(99))
        z = sc_divide(x, y)
        np.testing.assert_allclose(np.mean(z), 0.5, atol=0.05)

    def test_equal_numerator_denominator(self) -> None:
        """Equal densities give an output closer to 1.0 than to 0.0.

        These inputs are independent, so the circuit gives p / (2p - p^2), about
        0.714 for p = 0.6, not 1; correlated inputs give 1.
        """
        L = 8192
        p = 0.6
        x = generate_bernoulli_bitstream(p, L, rng=RNG(42))
        y = generate_bernoulli_bitstream(p, L, rng=RNG(99))
        z = sc_divide(x, y)
        result = np.mean(z)
        assert result > 0.5, f"p/p gave {result:.3f}, expected > 0.5"

    def test_quarter_by_half(self) -> None:
        """0.25 / 0.5 should be in the right neighbourhood.

        These inputs are independent, so the circuit gives 0.25 / 0.625 = 0.4,
        not 0.5; correlated inputs give the quotient.
        """
        L = 8192
        x = generate_bernoulli_bitstream(0.25, L, rng=RNG(1))
        y = generate_bernoulli_bitstream(0.5, L, rng=RNG(2))
        z = sc_divide(x, y)
        result = np.mean(z)
        assert 0.2 < result < 0.8, f"0.25/0.5 gave {result:.3f}, outside [0.2, 0.8]"

    def test_correlated_streams_give_the_quotient(self) -> None:
        """Streams compared against one shared random sequence give P(x) / P(y)."""
        shared = np.random.default_rng(20261117).random(1 << 16)
        x = (shared < 0.3).astype(np.uint8)
        y = (shared < 0.6).astype(np.uint8)
        assert np.all(y[x == 1] == 1)
        np.testing.assert_allclose(np.mean(sc_divide(x, y)), 0.5, atol=0.02)

    def test_independent_streams_give_a_different_fraction(self) -> None:
        """Independent streams give P(x) / (P(x) + P(y) - P(x) P(y)), not P(x) / P(y)."""
        rng = np.random.default_rng(1117)
        x = (rng.random(1 << 16) < 0.3).astype(np.uint8)
        y = (rng.random(1 << 16) < 0.6).astype(np.uint8)
        result = np.mean(sc_divide(x, y))
        np.testing.assert_allclose(result, 0.3 / (0.3 + 0.6 - 0.3 * 0.6), atol=0.02)
        assert abs(result - 0.5) > 0.05

    def test_output_binary(self) -> None:
        """Output should be binary (0 or 1)."""
        L = 1024
        x = generate_bernoulli_bitstream(0.3, L, rng=RNG(10))
        y = generate_bernoulli_bitstream(0.6, L, rng=RNG(20))
        z = sc_divide(x, y)
        assert set(np.unique(z)).issubset({0, 1})

    def test_output_length(self) -> None:
        """The output has the length of its inputs."""
        L = 512
        x = generate_bernoulli_bitstream(0.4, L, rng=RNG(10))
        y = generate_bernoulli_bitstream(0.8, L, rng=RNG(20))
        z = sc_divide(x, y)
        assert len(z) == L

    def test_monotonicity(self) -> None:
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

    def test_convergence_with_length(self) -> None:
        """The error against the independent-stream limit falls with the length.

        These inputs are independent, so the limit is P(x) / (P(x) + P(y) - P(x) P(y)),
        about 0.417, not the quotient 0.5; measured against 0.5 the error stays near 0.08.
        """
        pn, pd = 0.3, 0.6
        expected = pn / (pn + pd - pn * pd)
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
        assert errors[-1] < 0.02, f"L=4096 is {errors[-1]:.3f} from the limit"
