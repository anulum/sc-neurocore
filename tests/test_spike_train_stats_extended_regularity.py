# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegularity from former test_spike_train_stats_extended.py

"""Focused suite: TestRegularity from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403

class TestRegularity:
    def test_lvr_regular(self, regular_train):
        val = lvr(regular_train)
        assert val < 0.5

    def test_lvr_nan_empty(self):
        assert np.isnan(lvr(np.zeros(10)))

    def test_complexity_pdf(self, poisson_train):
        pdf = complexity_pdf(poisson_train)
        assert pdf.size > 0
        assert np.all(pdf >= 0)

    def test_optimal_bin_width(self, poisson_train):
        bw = optimal_bin_width(poisson_train)
        assert bw > 0

    def test_optimal_kernel_bandwidth(self, poisson_train):
        h = optimal_kernel_bandwidth(poisson_train)
        assert h > 0

    def test_lempel_ziv(self, poisson_train):
        c = lempel_ziv_complexity(poisson_train)
        assert c > 0

    def test_lempel_ziv_constant(self):
        assert lempel_ziv_complexity(np.zeros(100)) > 0

    def test_approximate_entropy(self, poisson_train):
        ae = approximate_entropy(poisson_train[:500])
        assert np.isfinite(ae)

    def test_sample_entropy(self, poisson_train):
        se = sample_entropy(poisson_train[:500])
        assert np.isfinite(se)

    def test_permutation_entropy_regular(self, regular_train):
        pe = permutation_entropy(regular_train, order=3, delay=1)
        assert 0.0 <= pe <= 1.0

    def test_hurst_exponent(self, poisson_train):
        h = hurst_exponent(poisson_train)
        assert 0.0 < h < 2.0

    def test_allan_factor(self, poisson_train):
        af, windows = allan_factor(poisson_train)
        assert af.size == windows.size
        assert af.size > 0

    def test_rescaled_range(self, poisson_train):
        h = rescaled_range(poisson_train)
        assert np.isfinite(h)
