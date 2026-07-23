# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorrelationViolation from former test_sc_convergence.py

"""Focused suite: TestCorrelationViolation from former test_sc_convergence.py."""

from __future__ import annotations

from tests.sc_convergence_support import *  # noqa: F403

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
        assert corr_err > ind_err, "correlated inputs should produce larger error than independent"
