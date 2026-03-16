# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for spike train analysis toolkit."""

from __future__ import annotations

import numpy as np

from sc_neurocore.analysis.spike_train_stats import (
    spike_times,
    isi,
    firing_rate,
    cv_isi,
    fano_factor,
    spike_count,
    psth,
    cross_correlation,
    pairwise_correlation,
    power_spectrum,
    burst_detection,
)


def _poisson_train(rate_hz: float, duration_s: float, dt: float = 0.001, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    return (rng.random(n) < rate_hz * dt).astype(np.uint8)


class TestSpikeTimes:
    def test_basic(self):
        train = np.array([0, 1, 0, 0, 1, 0], dtype=np.uint8)
        t = spike_times(train, dt=0.001)
        assert len(t) == 2
        np.testing.assert_allclose(t, [0.001, 0.004])

    def test_empty(self):
        assert spike_times(np.zeros(10, dtype=np.uint8)).size == 0


class TestISI:
    def test_regular(self):
        train = np.zeros(100, dtype=np.uint8)
        train[10::20] = 1
        intervals = isi(train, dt=0.001)
        np.testing.assert_allclose(intervals, 0.02, atol=1e-10)

    def test_single_spike(self):
        train = np.zeros(50, dtype=np.uint8)
        train[25] = 1
        assert isi(train).size == 0


class TestFiringRate:
    def test_known_rate(self):
        train = _poisson_train(100.0, 1.0)
        rate = firing_rate(train, dt=0.001)
        assert 70 < rate < 140

    def test_empty(self):
        assert firing_rate(np.zeros(100, dtype=np.uint8)) == 0.0


class TestCVISI:
    def test_regular_low_cv(self):
        train = np.zeros(1000, dtype=np.uint8)
        train[10::20] = 1
        assert cv_isi(train) < 0.05

    def test_poisson_near_one(self):
        train = _poisson_train(50.0, 5.0)
        cv = cv_isi(train)
        assert 0.5 < cv < 1.5

    def test_too_few_spikes(self):
        train = np.zeros(100, dtype=np.uint8)
        train[50] = 1
        assert np.isnan(cv_isi(train))


class TestFanoFactor:
    def test_poisson_near_one(self):
        train = _poisson_train(100.0, 5.0)
        ff = fano_factor(train, window_ms=100.0)
        assert 0.5 < ff < 2.0

    def test_regular_below_one(self):
        train = np.zeros(5000, dtype=np.uint8)
        train[10::20] = 1
        ff = fano_factor(train, window_ms=100.0)
        assert ff < 0.5


class TestSpikeCount:
    def test_count(self):
        train = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        assert spike_count(train) == 4


class TestPSTH:
    def test_shape(self):
        trials = [_poisson_train(100.0, 0.5, seed=i) for i in range(10)]
        rates, centers = psth(trials, bin_ms=10.0)
        assert rates.size > 0
        assert rates.size == centers.size

    def test_empty(self):
        rates, centers = psth([])
        assert rates.size == 0


class TestCrossCorrelation:
    def test_autocorrelation_peak_at_zero(self):
        train = _poisson_train(100.0, 1.0)
        cc, lags = cross_correlation(train, train, max_lag_ms=20.0)
        zero_idx = len(lags) // 2
        assert cc[zero_idx] == cc.max()

    def test_independent_low_correlation(self):
        a = _poisson_train(100.0, 1.0, seed=1)
        b = _poisson_train(100.0, 1.0, seed=2)
        cc, _ = cross_correlation(a, b, max_lag_ms=10.0)
        assert np.abs(cc).max() < 0.3


class TestPairwiseCorrelation:
    def test_self_correlation(self):
        train = _poisson_train(100.0, 1.0)
        mat = pairwise_correlation([train, train])
        np.testing.assert_allclose(mat[0, 1], 1.0, atol=1e-10)

    def test_shape(self):
        trains = [_poisson_train(50.0, 0.5, seed=i) for i in range(5)]
        mat = pairwise_correlation(trains)
        assert mat.shape == (5, 5)


class TestPowerSpectrum:
    def test_has_values(self):
        train = _poisson_train(100.0, 1.0)
        psd, freqs = power_spectrum(train)
        assert psd.size > 0
        assert freqs.size == psd.size
        assert np.all(psd >= 0)

    def test_empty(self):
        psd, freqs = power_spectrum(np.array([0], dtype=np.uint8))
        assert psd.size == 0


class TestBurstDetection:
    def test_detects_burst(self):
        train = np.zeros(1000, dtype=np.uint8)
        train[100:106] = 1
        bursts = burst_detection(train, dt=0.001, max_isi_ms=2.0, min_spikes=3)
        assert len(bursts) >= 1
        assert bursts[0][2] >= 3

    def test_no_burst_in_regular(self):
        train = np.zeros(1000, dtype=np.uint8)
        train[::50] = 1
        bursts = burst_detection(train, dt=0.001, max_isi_ms=5.0, min_spikes=3)
        assert len(bursts) == 0
