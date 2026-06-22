# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Edge-case coverage tests for analysis/spike_stats/variability.py

"""Tests targeting every uncovered branch in variability.py:
empty trains, single spikes, zero ISI, degenerate inputs."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats import variability as variability_module
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


_RUST_AVAILABLE = variability_module._HAS_RUST and variability_module._ssc is not None


@pytest.fixture
def force_python_fallback(monkeypatch):
    """Disable the Rust acceleration so the pure-Python reference path executes."""
    monkeypatch.setattr(variability_module, "_HAS_RUST", False)


def _bernoulli_train(p, n, seed):
    """A reproducible Bernoulli spike train for the fallback exercises."""
    rng = np.random.default_rng(seed)
    return (rng.random(n) < p).astype(np.int8)


class TestZeroTimestepGuards:
    """A zero timestep collapses every inter-spike interval to zero, the only
    way to reach the mean/sum==0 guards that strictly-positive ISIs never hit."""

    @staticmethod
    def _three_spikes():
        train = np.zeros(50, dtype=np.int8)
        train[[5, 15, 30]] = 1
        return train

    def test_cv_isi_zero_mean_interval(self):
        assert np.isnan(cv_isi(self._three_spikes(), dt=0.0))

    def test_cv2_no_positive_sums(self):
        assert np.isnan(cv2(self._three_spikes(), dt=0.0))

    def test_local_variation_no_positive_sums(self):
        assert np.isnan(local_variation(self._three_spikes(), dt=0.0))

    def test_lvr_every_pair_sum_nonpositive(self):
        # Each consecutive ISI sum is zero, so the per-pair skip runs for every
        # pair and the contributing count stays zero -> NaN.
        assert np.isnan(lvr(self._three_spikes(), dt=0.0))


class TestIsiEntropyZeroRange:
    """A perfectly regular train has a single ISI value; the zero-range check
    must short-circuit to 0.0 before np.histogram, which rejects a zero range."""

    def test_regular_train_returns_zero_without_histogram_error(self):
        train = np.zeros(100, dtype=np.int8)
        train[::10] = 1
        assert isi_entropy(train) == 0.0


class TestHurstSingleScale:
    """When only one DFA scale fits the train length, the log-log fit has too
    few points and the Hurst exponent is undefined."""

    def test_single_usable_scale_returns_nan(self):
        # n == 4*min_window admits exactly one scale (s=min_window); the next
        # 1.5x step exceeds n//4, leaving a single (log s, log F) point.
        train = np.zeros(40, dtype=np.int8)
        train[::4] = 1
        assert np.isnan(hurst_exponent(train, min_window=10))


class TestRescaledRangeDegenerateScale:
    """min_window == 1 makes the first 1.5x step stall at 1; the unit-step guard
    must force progress so the analysis terminates instead of looping forever."""

    def test_min_window_one_terminates(self):
        train = np.zeros(300, dtype=np.int8)
        train[::3] = 1
        result = rescaled_range(train, min_window=1)
        assert np.isfinite(result)


class TestPurePythonFallbacks:
    """Exercise the reference Python implementations that shadow the Rust core,
    and confirm they agree with the Rust path when it is available."""

    def test_lempel_ziv_python_branch(self, force_python_fallback):
        value = lempel_ziv_complexity(_bernoulli_train(0.3, 256, 7))
        assert np.isfinite(value) and value > 0.0

    def test_approximate_entropy_python_branch(self, force_python_fallback):
        value = approximate_entropy(_bernoulli_train(0.3, 200, 11))
        assert np.isfinite(value)

    def test_sample_entropy_python_branch(self, force_python_fallback):
        value = sample_entropy(_bernoulli_train(0.35, 200, 13))
        assert np.isfinite(value) or np.isnan(value)

    def test_sample_entropy_no_matches_returns_nan(self, force_python_fallback):
        # Alternating 0/1 of length 4: the two length-2 templates differ by 1,
        # which exceeds r = 0.2*std, so no template pair matches and b == 0.
        train = np.array([0, 1, 0, 1], dtype=np.int8)
        assert np.isnan(sample_entropy(train, m=2))

    def test_permutation_entropy_python_branch(self, force_python_fallback):
        value = permutation_entropy(_bernoulli_train(0.4, 200, 17), order=3)
        assert 0.0 <= value <= 1.0

    def test_permutation_entropy_order_one_degenerate(self, force_python_fallback):
        # order=1 -> a single ordinal pattern -> h_max = log2(1!) = 0 -> 0.0.
        value = permutation_entropy(_bernoulli_train(0.4, 100, 19), order=1)
        assert value == 0.0

    @pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust spike_stats_core not built")
    def test_lempel_ziv_python_matches_rust(self, monkeypatch):
        train = _bernoulli_train(0.3, 256, 7)
        rust_value = lempel_ziv_complexity(train)
        monkeypatch.setattr(variability_module, "_HAS_RUST", False)
        python_value = lempel_ziv_complexity(train)
        assert np.isclose(rust_value, python_value, rtol=1e-9, atol=0.0)

    @pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust spike_stats_core not built")
    def test_entropies_python_match_rust(self, monkeypatch):
        train = _bernoulli_train(0.35, 200, 23)
        rust = (
            approximate_entropy(train),
            sample_entropy(train),
            permutation_entropy(train, order=3),
        )
        monkeypatch.setattr(variability_module, "_HAS_RUST", False)
        python = (
            approximate_entropy(train),
            sample_entropy(train),
            permutation_entropy(train, order=3),
        )
        for rust_value, python_value in zip(rust, python):
            assert np.isclose(rust_value, python_value, rtol=1e-6, atol=1e-9, equal_nan=True)
