# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestANDMultiplicationConvergence from former test_sc_convergence.py

"""Focused suite: TestANDMultiplicationConvergence from former test_sc_convergence.py."""

from __future__ import annotations

from tests.sc_convergence_support import *  # noqa: F403


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
