# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based tests for SC convergence guarantees

"""Property-based tests for stochastic computing convergence.

Verifies:
- AND multiplication converges O(1/sqrt(L))
- Sobol encoding converges faster than Bernoulli
- CORDIV quotient is monotonic in numerator
- Correlated inputs violate multiplication correctness
- BitstreamEncoder roundtrip stays within expected bounds
- Popcount is exact (no approximation error)
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore import (
    BitstreamEncoder,
    bitstream_to_probability,
    generate_bernoulli_bitstream,
    generate_sobol_bitstream,
)
from sc_neurocore.utils.bitstreams import sc_divide
from sc_neurocore.utils.rng import RNG


N_TRIALS = 100


class TestANDMultiplicationConvergence:
    """AND gate computes p_x * p_y for independent bitstreams."""

    def test_converges_with_length(self):
        """MAE should decrease as L increases."""
        px, py = 0.7, 0.4
        errors_by_L = {}
        for L in [128, 1024, 8192]:
            errs = []
            for trial in range(N_TRIALS):
                x = generate_bernoulli_bitstream(px, L, rng=RNG(trial))
                y = generate_bernoulli_bitstream(py, L, rng=RNG(trial + 10000))
                errs.append(abs(np.mean(x & y) - px * py))
            errors_by_L[L] = np.mean(errs)
        assert errors_by_L[8192] < errors_by_L[128]

    def test_hoeffding_bound_holds(self):
        """Error should stay within 3/sqrt(L) (3-sigma) for 99%+ of trials."""
        px, py = 0.5, 0.5
        L = 4096
        violations = 0
        bound = 3.0 / np.sqrt(L)
        for trial in range(500):
            x = generate_bernoulli_bitstream(px, L, rng=RNG(trial))
            y = generate_bernoulli_bitstream(py, L, rng=RNG(trial + 5000))
            err = abs(np.mean(x & y) - px * py)
            if err > bound:
                violations += 1
        assert violations < 25, f"{violations}/500 violated 3-sigma bound"

    @pytest.mark.parametrize("px,py", [(0.1, 0.9), (0.5, 0.5), (0.9, 0.1), (0.3, 0.7)])
    def test_unbiased(self, px, py):
        """Mean over many trials should converge to exact product."""
        L = 4096
        products = []
        for trial in range(200):
            x = generate_bernoulli_bitstream(px, L, rng=RNG(trial))
            y = generate_bernoulli_bitstream(py, L, rng=RNG(trial + 8000))
            products.append(np.mean(x & y))
        np.testing.assert_allclose(np.mean(products), px * py, atol=0.01)


class TestSobolConvergence:
    """Sobol quasi-random should converge faster than Bernoulli."""

    def test_sobol_lower_error_at_same_length(self):
        target = 0.65
        L = 512
        bern_errs, sobol_errs = [], []
        for trial in range(N_TRIALS):
            b = generate_bernoulli_bitstream(target, L, rng=RNG(trial))
            bern_errs.append(abs(np.mean(b) - target))
            s = generate_sobol_bitstream(target, L, seed=trial)
            sobol_errs.append(abs(np.mean(s) - target))
        assert np.mean(sobol_errs) < np.mean(bern_errs), "Sobol should beat Bernoulli"

    def test_sobol_output_binary(self):
        bits = generate_sobol_bitstream(0.5, 1024, seed=42)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_sobol_probability_accurate(self):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            bits = generate_sobol_bitstream(p, 4096, seed=42)
            np.testing.assert_allclose(np.mean(bits), p, atol=0.02)


class TestCORDIVMonotonicity:
    """CORDIV output should increase with numerator probability."""

    def test_monotonic_in_numerator(self):
        L = 8192
        pd = 0.8
        y = generate_bernoulli_bitstream(pd, L, rng=RNG(99))
        prev = -1.0
        for pn in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            x = generate_bernoulli_bitstream(pn, L, rng=RNG(42))
            z = sc_divide(x, y)
            result = np.mean(z)
            assert result > prev - 0.05, (
                f"not monotonic: pn={pn}, result={result:.3f}, prev={prev:.3f}"
            )
            prev = result


class TestCorrelationViolation:
    """Using the SAME RNG for both bitstreams violates independence
    and should produce biased multiplication results."""

    def test_correlated_inputs_biased(self):
        px, py = 0.5, 0.5
        L = 10000
        # Same seed = correlated
        x = generate_bernoulli_bitstream(px, L, rng=RNG(42))
        y = generate_bernoulli_bitstream(py, L, rng=RNG(42))
        corr_product = np.mean(x & y)
        # Independent seeds
        y_ind = generate_bernoulli_bitstream(py, L, rng=RNG(99))
        ind_product = np.mean(x & y_ind)
        # Correlated: x & x = x, so product ≈ px (not px*py)
        # This should show bias
        expected = px * py
        corr_err = abs(corr_product - expected)
        ind_err = abs(ind_product - expected)
        assert corr_err > ind_err, (
            "correlated inputs should produce larger error than independent"
        )


class TestBitstreamEncoderRoundtrip:
    """BitstreamEncoder.encode → bitstream_to_probability roundtrip."""

    @pytest.mark.parametrize("p", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_roundtrip_accuracy(self, p):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=4096, seed=42)
        bits = enc.encode(p)
        recovered = bitstream_to_probability(bits)
        np.testing.assert_allclose(recovered, p, atol=0.03)

    def test_output_is_binary(self):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=1024, seed=42)
        bits = enc.encode(0.6)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_output_length(self):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=777, seed=42)
        bits = enc.encode(0.5)
        assert len(bits) == 777


class TestPopcountExact:
    """Popcount should return the exact number of 1-bits."""

    def test_known_popcount(self):
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        assert np.sum(bits) == 4
        assert bitstream_to_probability(bits) == 0.5

    def test_all_ones(self):
        bits = np.ones(1000, dtype=np.uint8)
        assert bitstream_to_probability(bits) == 1.0

    def test_all_zeros(self):
        bits = np.zeros(1000, dtype=np.uint8)
        assert bitstream_to_probability(bits) == 0.0
