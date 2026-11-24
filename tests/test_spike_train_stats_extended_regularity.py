# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegularity from former test_spike_train_stats_extended.py

"""Focused suite: TestRegularity from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import allan_factor
from sc_neurocore.analysis.spike_stats import approximate_entropy
from sc_neurocore.analysis.spike_stats import complexity_pdf
from sc_neurocore.analysis.spike_stats import hurst_exponent
from sc_neurocore.analysis.spike_stats import lempel_ziv_complexity
from sc_neurocore.analysis.spike_stats import lvr
from sc_neurocore.analysis.spike_stats import optimal_bin_width
from sc_neurocore.analysis.spike_stats import optimal_kernel_bandwidth
from sc_neurocore.analysis.spike_stats import permutation_entropy
from sc_neurocore.analysis.spike_stats import rescaled_range
from sc_neurocore.analysis.spike_stats import sample_entropy
import numpy as np
from tests.spike_train_stats_extended_support import poisson_train as poisson_train
from tests.spike_train_stats_extended_support import regular_train as regular_train


class TestRegularity:
    """Regularity, complexity and long-range dependence of spike trains."""

    def test_lvr_regular(self, regular_train: np.ndarray[Any, Any]) -> None:
        """The revised local variation of a regular train is below 0.5."""
        val = lvr(regular_train)
        assert val < 0.5

    def test_lvr_nan_empty(self) -> None:
        """The revised local variation of a silent train is NaN."""
        assert np.isnan(lvr(np.zeros(10)))

    def test_complexity_pdf(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The complexity distribution of a Poisson train is not empty and non-negative."""
        pdf = complexity_pdf(poisson_train)
        assert pdf.size > 0
        assert np.all(pdf >= 0)

    def test_optimal_bin_width(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The optimal bin width of a Poisson train is positive."""
        bw = optimal_bin_width(poisson_train)
        assert bw > 0

    def test_optimal_kernel_bandwidth(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The optimal kernel bandwidth of a Poisson train is positive."""
        h = optimal_kernel_bandwidth(poisson_train)
        assert h > 0

    def test_lempel_ziv(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The Lempel-Ziv complexity of a Poisson train is positive."""
        c = lempel_ziv_complexity(poisson_train)
        assert c > 0

    def test_lempel_ziv_constant(self) -> None:
        """The Lempel-Ziv complexity of a silent train is positive."""
        assert lempel_ziv_complexity(np.zeros(100)) > 0

    def test_approximate_entropy(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """Approximate entropy of 500 Poisson samples is finite."""
        ae = approximate_entropy(poisson_train[:500])
        assert np.isfinite(ae)

    def test_sample_entropy(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """Sample entropy of 500 Poisson samples is finite."""
        se = sample_entropy(poisson_train[:500])
        assert np.isfinite(se)

    def test_permutation_entropy_regular(self, regular_train: np.ndarray[Any, Any]) -> None:
        """Permutation entropy of a regular train lies inside [0, 1]."""
        pe = permutation_entropy(regular_train, order=3, delay=1)
        assert 0.0 <= pe <= 1.0

    def test_hurst_exponent(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The Hurst exponent of a Poisson train lies strictly between 0 and 2."""
        h = hurst_exponent(poisson_train)
        assert 0.0 < h < 2.0

    def test_allan_factor(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The Allan factor has one value per window and is not empty."""
        af, windows = allan_factor(poisson_train)
        assert af.size == windows.size
        assert af.size > 0

    def test_rescaled_range(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The rescaled range of a Poisson train is finite."""
        h = rescaled_range(poisson_train)
        assert np.isfinite(h)
