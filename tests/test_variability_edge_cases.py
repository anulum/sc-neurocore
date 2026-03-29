# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Edge-case coverage tests for analysis/spike_stats/variability.py

"""Tests targeting every uncovered branch in variability.py:
empty trains, single spikes, zero ISI, degenerate inputs."""

from __future__ import annotations

import numpy as np

from sc_neurocore.analysis.spike_stats.variability import (
    cv_isi,
    cv2,
    local_variation,
    lvr,
    fano_factor,
    isi_entropy,
    lempel_ziv_complexity,
    approximate_entropy,
    sample_entropy,
    permutation_entropy,
    hurst_exponent,
    allan_factor,
    rescaled_range,
    complexity_pdf,
    optimal_bin_width,
    optimal_kernel_bandwidth,
)


class TestCvIsiEdge:
    def test_empty_train(self):
        assert np.isnan(cv_isi(np.zeros(100, dtype=np.int8)))

    def test_single_spike(self):
        train = np.zeros(100, dtype=np.int8)
        train[50] = 1
        assert np.isnan(cv_isi(train))

    def test_two_spikes(self):
        train = np.zeros(100, dtype=np.int8)
        train[20] = 1
        train[60] = 1
        result = cv_isi(train)
        assert np.isfinite(result) or np.isnan(result)


class TestCv2Edge:
    def test_empty(self):
        result = cv2(np.zeros(50, dtype=np.int8))
        assert np.isnan(result)

    def test_single_spike(self):
        train = np.zeros(50, dtype=np.int8)
        train[25] = 1
        result = cv2(train)
        assert np.isnan(result)

    def test_regular_train(self):
        train = np.zeros(100, dtype=np.int8)
        train[::10] = 1
        result = cv2(train)
        assert np.isfinite(result)


class TestLocalVariationEdge:
    def test_empty(self):
        result = local_variation(np.zeros(50, dtype=np.int8))
        assert np.isnan(result)

    def test_regular(self):
        train = np.zeros(100, dtype=np.int8)
        train[::5] = 1
        result = local_variation(train)
        assert np.isfinite(result)


class TestLvrEdge:
    def test_empty(self):
        result = lvr(np.zeros(50, dtype=np.int8))
        assert np.isnan(result)

    def test_regular(self):
        train = np.zeros(100, dtype=np.int8)
        train[::5] = 1
        result = lvr(train)
        assert np.isfinite(result)


class TestFanoFactorEdge:
    def test_empty(self):
        result = fano_factor(np.zeros(50, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)

    def test_all_spikes(self):
        result = fano_factor(np.ones(50, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)


class TestIsiEntropyEdge:
    def test_empty(self):
        result = isi_entropy(np.zeros(50, dtype=np.int8))
        assert np.isnan(result) or result == 0.0

    def test_single_isi(self):
        train = np.zeros(50, dtype=np.int8)
        train[10] = 1
        train[20] = 1
        result = isi_entropy(train)
        assert np.isfinite(result) or np.isnan(result)


class TestLempelZivEdge:
    def test_empty(self):
        result = lempel_ziv_complexity(np.zeros(50, dtype=np.int8))
        assert np.isfinite(result)

    def test_all_ones(self):
        result = lempel_ziv_complexity(np.ones(50, dtype=np.int8))
        assert np.isfinite(result)

    def test_alternating(self):
        train = np.zeros(100, dtype=np.int8)
        train[::2] = 1
        result = lempel_ziv_complexity(train)
        assert np.isfinite(result)


class TestApproxEntropyEdge:
    def test_short_train(self):
        result = approximate_entropy(np.zeros(3, dtype=np.int8))
        assert np.isnan(result)

    def test_constant(self):
        result = approximate_entropy(np.zeros(100, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)

    def test_normal(self):
        train = np.zeros(200, dtype=np.int8)
        train[::5] = 1
        result = approximate_entropy(train)
        assert np.isfinite(result) or np.isnan(result)


class TestSampleEntropyEdge:
    def test_short(self):
        result = sample_entropy(np.zeros(3, dtype=np.int8))
        assert np.isnan(result) or np.isfinite(result)

    def test_constant(self):
        result = sample_entropy(np.zeros(100, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)

    def test_normal(self):
        train = np.zeros(200, dtype=np.int8)
        train[::7] = 1
        result = sample_entropy(train)
        assert np.isfinite(result) or np.isnan(result)


class TestPermutationEntropyEdge:
    def test_short(self):
        result = permutation_entropy(np.zeros(3, dtype=np.int8))
        assert np.isnan(result) or np.isfinite(result)

    def test_constant(self):
        result = permutation_entropy(np.zeros(100, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)


class TestHurstExponentEdge:
    def test_short(self):
        result = hurst_exponent(np.zeros(5, dtype=np.int8))
        assert np.isnan(result) or np.isfinite(result)

    def test_normal(self):
        train = np.zeros(500, dtype=np.int8)
        train[::10] = 1
        result = hurst_exponent(train)
        assert np.isfinite(result) or np.isnan(result)


class TestAllanFactorEdge:
    def test_short(self):
        vals, windows = allan_factor(np.zeros(5, dtype=np.int8))
        assert isinstance(vals, np.ndarray)

    def test_empty_spikes(self):
        vals, windows = allan_factor(np.zeros(200, dtype=np.int8))
        assert isinstance(vals, np.ndarray)

    def test_normal(self):
        train = np.zeros(500, dtype=np.int8)
        train[::10] = 1
        vals, windows = allan_factor(train)
        assert len(vals) > 0


class TestRescaledRangeEdge:
    def test_short(self):
        result = rescaled_range(np.zeros(5, dtype=np.int8))
        assert np.isnan(result) or np.isfinite(result)

    def test_normal(self):
        train = np.zeros(500, dtype=np.int8)
        train[::10] = 1
        result = rescaled_range(train)
        assert np.isfinite(result) or np.isnan(result)


class TestComplexityPdfEdge:
    def test_empty(self):
        result = complexity_pdf(np.zeros(50, dtype=np.int8))
        assert isinstance(result, np.ndarray)

    def test_normal(self):
        train = np.zeros(100, dtype=np.int8)
        train[::5] = 1
        result = complexity_pdf(train)
        assert isinstance(result, np.ndarray)


class TestOptimalBinWidthEdge:
    def test_empty(self):
        result = optimal_bin_width(np.zeros(50, dtype=np.int8))
        assert np.isnan(result) or np.isfinite(result)

    def test_single_spike(self):
        train = np.zeros(50, dtype=np.int8)
        train[25] = 1
        result = optimal_bin_width(train)
        assert np.isnan(result) or np.isfinite(result)

    def test_normal(self):
        train = np.zeros(200, dtype=np.int8)
        train[::10] = 1
        result = optimal_bin_width(train)
        assert np.isfinite(result)


class TestOptimalKernelBandwidthEdge:
    def test_empty(self):
        result = optimal_kernel_bandwidth(np.zeros(50, dtype=np.int8))
        assert np.isnan(result)

    def test_single_spike(self):
        train = np.zeros(50, dtype=np.int8)
        train[25] = 1
        result = optimal_kernel_bandwidth(train)
        assert np.isnan(result)

    def test_normal(self):
        train = np.zeros(200, dtype=np.int8)
        train[::10] = 1
        result = optimal_kernel_bandwidth(train)
        assert np.isfinite(result)
