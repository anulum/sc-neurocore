# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for spike train analysis toolkit

"""Tests for spike train analysis toolkit."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats import (
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
    instantaneous_rate,
    van_rossum_distance,
    victor_purpura_distance,
    isi_distance,
    cv2,
    local_variation,
    isi_entropy,
    event_synchronization,
    spike_train_coherence,
    first_spike_latency,
    response_onset,
    spike_triggered_average,
    bin_spike_train,
    population_rate,
    surrogate_isi_shuffle,
    surrogate_dither,
    surrogate_trial_shuffle,
    mutual_information,
    transfer_entropy,
    phase_locking_value,
    spike_field_coherence,
    spike_phase_histogram,
    spike_train_pca,
    population_vector_decode,
    functional_connectivity,
    significance_bootstrap,
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

    def test_silent_train_returns_zero_correlogram(self) -> None:
        # A silent train has zero variance, so the normaliser is zero and the
        # correlogram is returned flat rather than dividing by zero.
        silent = np.zeros(200, dtype=np.float64)
        cc, lags = cross_correlation(silent, silent, max_lag_ms=10.0)
        assert np.all(cc == 0.0)
        assert cc.size == lags.size


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


class TestInstantaneousRate:
    def test_gaussian(self):
        train = _poisson_train(100.0, 1.0)
        rate = instantaneous_rate(train, kernel="gaussian", sigma_ms=20.0)
        assert rate.size == train.size
        assert rate.mean() > 0

    def test_exponential(self):
        rate = instantaneous_rate(_poisson_train(50.0, 0.5), kernel="exponential")
        assert rate.size > 0

    def test_rectangular(self):
        rate = instantaneous_rate(_poisson_train(50.0, 0.5), kernel="rectangular")
        assert rate.size > 0


class TestVanRossumDistance:
    def test_identical_zero(self):
        train = _poisson_train(100.0, 0.5)
        d = van_rossum_distance(train, train)
        assert d < 1e-6

    def test_different_positive(self):
        a = _poisson_train(100.0, 0.5, seed=1)
        b = _poisson_train(100.0, 0.5, seed=2)
        d = van_rossum_distance(a, b)
        assert d > 0


class TestVictorPurpuraDistance:
    def test_identical_zero(self):
        t = np.array([0.1, 0.2, 0.3])
        assert victor_purpura_distance(t, t) < 1e-10

    def test_empty(self):
        assert victor_purpura_distance(np.array([]), np.array([0.1, 0.2])) == 2.0

    def test_different(self):
        a = np.array([0.1, 0.3, 0.5])
        b = np.array([0.15, 0.35, 0.55])
        d = victor_purpura_distance(a, b, cost_per_s=100.0)
        assert d > 0


class TestISIDistance:
    def test_same_train(self):
        train = np.zeros(1000, dtype=np.uint8)
        train[10::20] = 1
        d = isi_distance(train, train)
        assert d < 1e-10

    def test_different(self):
        a = _poisson_train(50.0, 1.0, seed=1)
        b = _poisson_train(100.0, 1.0, seed=2)
        d = isi_distance(a, b)
        assert d > 0


class TestCV2:
    def test_regular_low(self):
        train = np.zeros(1000, dtype=np.uint8)
        train[10::20] = 1
        assert cv2(train) < 0.1

    def test_poisson(self):
        c = cv2(_poisson_train(50.0, 5.0))
        assert 0.3 < c < 1.5


class TestLocalVariation:
    def test_regular_low(self):
        train = np.zeros(1000, dtype=np.uint8)
        train[10::20] = 1
        lv = local_variation(train)
        assert lv < 0.1

    def test_poisson_near_one(self):
        lv = local_variation(_poisson_train(50.0, 5.0))
        assert 0.5 < lv < 1.5


class TestISIEntropy:
    def test_regular_low_entropy(self):
        train = np.zeros(2000, dtype=np.uint8)
        train[10::20] = 1
        h = isi_entropy(train)
        assert h < 2.0

    def test_poisson_higher(self):
        h = isi_entropy(_poisson_train(50.0, 5.0))
        assert h > 0


class TestEventSynchronization:
    def test_identical_high(self):
        train = _poisson_train(50.0, 0.5)
        s = event_synchronization(train, train, tau_ms=2.0)
        assert s > 0.5

    def test_independent_low(self):
        a = _poisson_train(50.0, 0.5, seed=1)
        b = _poisson_train(50.0, 0.5, seed=2)
        s = event_synchronization(a, b, tau_ms=1.0)
        assert s < 0.5

    def test_pure_python_fallback_when_rust_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the pure-Python double-loop branch; the native accelerator is
        # built in this environment and would otherwise shadow it.
        monkeypatch.setattr("sc_neurocore.analysis.spike_stats.correlation._HAS_RUST", False)
        train = _poisson_train(50.0, 0.5)
        s = event_synchronization(train, train, tau_ms=2.0)
        assert s > 0.5


class TestSpikeTrainCoherence:
    def test_shape(self):
        train = _poisson_train(100.0, 0.5)
        coh, freqs = spike_train_coherence(train, train)
        assert coh.size == freqs.size
        assert coh.size > 0


class TestFirstSpikeLatency:
    def test_known(self):
        train = np.zeros(100, dtype=np.uint8)
        train[42] = 1
        assert abs(first_spike_latency(train) - 0.042) < 1e-10

    def test_no_spike(self):
        assert np.isnan(first_spike_latency(np.zeros(100, dtype=np.uint8)))


class TestResponseOnset:
    def test_detects(self):
        train = np.zeros(500, dtype=np.uint8)
        train[200:210] = 1
        onset = response_onset(train, baseline_steps=150)
        assert 0.15 < onset < 0.25

    def test_no_response(self):
        assert np.isnan(response_onset(np.zeros(200, dtype=np.uint8), baseline_steps=100))


class TestSpikeTriggeredAverage:
    def test_shape(self):
        stim = np.sin(np.linspace(0, 10 * np.pi, 1000))
        train = np.zeros(1000, dtype=np.uint8)
        train[100::100] = 1
        sta = spike_triggered_average(stim, train, window_steps=50)
        assert sta.shape == (50,)

    def test_no_spikes(self):
        sta = spike_triggered_average(np.ones(100), np.zeros(100, dtype=np.uint8))
        np.testing.assert_allclose(sta, 0.0)


class TestBinSpikeTrain:
    def test_basic(self):
        train = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        binned = bin_spike_train(train, bin_size=5)
        assert binned.tolist() == [2, 3]


class TestPopulationRate:
    def test_positive(self):
        trains = [_poisson_train(100.0, 0.5, seed=i) for i in range(10)]
        rate = population_rate(trains, sigma_ms=20.0)
        assert rate.size > 0
        assert rate.mean() > 0

    def test_empty(self):
        assert population_rate([]).size == 0


class TestSurrogateISIShuffle:
    def test_preserves_count(self):
        train = _poisson_train(100.0, 0.5)
        surr = surrogate_isi_shuffle(train, seed=1)
        assert abs(surr.sum() - train.sum()) <= 1

    def test_different_order(self):
        train = _poisson_train(100.0, 1.0)
        surr = surrogate_isi_shuffle(train, seed=7)
        assert not np.array_equal(train, surr)


class TestSurrogateDither:
    def test_preserves_count(self):
        train = _poisson_train(50.0, 0.5)
        surr = surrogate_dither(train, dither_ms=3.0, seed=1)
        assert abs(int(surr.sum()) - int(train.sum())) <= 5


class TestSurrogateTrialShuffle:
    def test_preserves_trials(self):
        trains = [_poisson_train(50.0, 0.2, seed=i) for i in range(5)]
        shuffled = surrogate_trial_shuffle(trains, seed=1)
        assert len(shuffled) == 5
        sums_orig = sorted(t.sum() for t in trains)
        sums_shuf = sorted(t.sum() for t in shuffled)
        assert sums_orig == sums_shuf


class TestMutualInformation:
    def test_self_positive(self):
        train = _poisson_train(100.0, 1.0)
        mi = mutual_information(train, train, bin_size=20)
        assert mi > 0

    def test_independent_low(self):
        a = _poisson_train(50.0, 1.0, seed=1)
        b = _poisson_train(50.0, 1.0, seed=99)
        mi = mutual_information(a, b, bin_size=20)
        mi_self = mutual_information(a, a, bin_size=20)
        assert mi < mi_self


class TestTransferEntropy:
    def test_nonnegative(self):
        a = _poisson_train(100.0, 1.0, seed=1)
        b = _poisson_train(100.0, 1.0, seed=2)
        te = transfer_entropy(a, b, bin_size=20)
        assert te >= 0


class TestPhaseLockingValue:
    def test_locked(self):
        lfp = np.sin(2 * np.pi * 10 * np.arange(10000) * 0.001)
        train = np.zeros(10000, dtype=np.uint8)
        peaks = np.where(lfp > 0.99)[0]
        train[peaks] = 1
        plv = phase_locking_value(train, lfp)
        assert plv > 0.5

    def test_random_low(self):
        lfp = np.sin(2 * np.pi * 10 * np.arange(5000) * 0.001)
        train = _poisson_train(50.0, 5.0)[:5000]
        plv = phase_locking_value(train, lfp)
        assert plv < 0.5


class TestSpikeFieldCoherence:
    def test_shape(self):
        train = _poisson_train(100.0, 0.5)
        lfp = np.sin(2 * np.pi * 40 * np.arange(train.size) * 0.001)
        sfc, freqs = spike_field_coherence(train, lfp)
        assert sfc.size == freqs.size
        assert sfc.size > 0


class TestSpikePhaseHistogram:
    def test_shape(self):
        lfp = np.sin(2 * np.pi * 10 * np.arange(5000) * 0.001)
        train = _poisson_train(100.0, 5.0)[:5000]
        hist, centers = spike_phase_histogram(train, lfp, n_bins=18)
        assert hist.size == 18
        assert centers.size == 18


class TestSpikeTrainPCA:
    def test_shape(self):
        trains = [_poisson_train(50.0 + i * 10, 0.5, seed=i) for i in range(8)]
        proj, var = spike_train_pca(trains, n_components=3)
        assert proj.shape[0] == 3
        assert var.size == 3
        assert np.all(var >= 0)


class TestPopulationVectorDecode:
    def test_shape(self):
        trains = [_poisson_train(50.0, 0.5, seed=i) for i in range(4)]
        dirs = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        decoded = population_vector_decode(trains, dirs, window=50)
        assert decoded.size > 0

    def test_empty(self):
        assert population_vector_decode([], np.array([])).size == 0


class TestFunctionalConnectivity:
    def test_symmetric(self):
        trains = [_poisson_train(50.0, 0.5, seed=i) for i in range(4)]
        mat = functional_connectivity(trains, max_lag_ms=10.0)
        np.testing.assert_allclose(mat, mat.T, atol=1e-12)
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(np.diag(mat), 1.0)


class TestSignificanceBootstrap:
    def test_returns_pvalue(self):
        a = _poisson_train(100.0, 0.5, seed=1)
        b = _poisson_train(100.0, 0.5, seed=2)

        def stat(x, y):
            return abs(x.mean() - y.mean())

        obs, pval = significance_bootstrap(stat, a, b, n_surrogates=50, seed=42)
        assert 0.0 <= pval <= 1.0
        assert obs >= 0
