# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage tests for analysis/spike_stats/sorting_quality.py

"""Edge-case tests for spike sorting quality metrics."""

from __future__ import annotations

import numpy as np

from sc_neurocore.analysis.spike_stats.sorting_quality import (
    isolation_distance,
    l_ratio,
    silhouette_score,
    d_prime,
    isi_violation_rate,
    presence_ratio,
    amplitude_cutoff,
    snr,
    nn_hit_rate,
    drift_metric,
)


def _rng():
    return np.random.default_rng(42)


class TestIsolationDistance:
    def test_typical(self):
        rng = _rng()
        cluster = rng.normal(0, 1, (20, 3))
        noise = rng.normal(5, 1, (30, 3))
        result = isolation_distance(cluster, noise)
        assert np.isfinite(result)

    def test_too_small_cluster(self):
        result = isolation_distance(np.array([[1, 2]]), np.array([[3, 4], [5, 6]]))
        assert np.isnan(result)

    def test_single_feature(self):
        cluster = _rng().normal(0, 1, (10, 1))
        noise = _rng().normal(3, 1, (20, 1))
        result = isolation_distance(cluster, noise)
        assert np.isfinite(result)


class TestLRatio:
    def test_typical(self):
        rng = _rng()
        cluster = rng.normal(0, 1, (15, 2))
        noise = rng.normal(3, 1, (25, 2))
        result = l_ratio(cluster, noise)
        assert np.isfinite(result)

    def test_small_cluster(self):
        result = l_ratio(np.array([[1, 2]]), np.array([[3, 4]]))
        assert np.isnan(result)

    def test_empty_noise(self):
        cluster = _rng().normal(0, 1, (10, 2))
        result = l_ratio(cluster, np.empty((0, 2)))
        assert np.isnan(result)

    def test_single_feature(self):
        cluster = _rng().normal(0, 1, (10, 1))
        noise = _rng().normal(3, 1, (20, 1))
        result = l_ratio(cluster, noise)
        assert np.isfinite(result)


class TestSilhouetteScore:
    def test_typical(self):
        rng = _rng()
        features = np.vstack([rng.normal(0, 1, (10, 2)), rng.normal(5, 1, (10, 2))])
        labels = np.array([0] * 10 + [1] * 10)
        result = silhouette_score(features, labels)
        assert -1 <= result <= 1

    def test_single_point(self):
        result = silhouette_score(np.array([[1, 2]]), np.array([0]))
        assert result == 0.0

    def test_single_class(self):
        features = _rng().normal(0, 1, (10, 2))
        labels = np.zeros(10, dtype=int)
        result = silhouette_score(features, labels)
        assert result == 0.0


class TestDPrime:
    def test_typical(self):
        rng = _rng()
        a = rng.normal(0, 1, (20, 3))
        b = rng.normal(3, 1, (20, 3))
        result = d_prime(a, b)
        assert result > 0

    def test_identical_clusters(self):
        data = _rng().normal(0, 1, (10, 2))
        result = d_prime(data, data.copy())
        assert result == 0.0

    def test_zero_variance(self):
        a = np.ones((5, 2))
        b = np.ones((5, 2)) * 2
        result = d_prime(a, b)
        assert result == 0.0 or np.isfinite(result)


class TestIsiViolationRate:
    def test_no_violations(self):
        train = np.zeros(1000, dtype=np.int8)
        train[::100] = 1  # 10 Hz, ISI = 100 ms >> 1.5 ms
        result = isi_violation_rate(train)
        assert result == 0.0

    def test_empty(self):
        result = isi_violation_rate(np.zeros(100, dtype=np.int8))
        assert result == 0.0

    def test_all_violations(self):
        train = np.ones(10, dtype=np.int8)  # ISI = 1 ms < 1.5 ms
        result = isi_violation_rate(train)
        assert result > 0


class TestPresenceRatio:
    def test_full_presence(self):
        train = np.zeros(1000, dtype=np.int8)
        train[::10] = 1
        result = presence_ratio(train)
        assert result > 0.5

    def test_no_spikes(self):
        result = presence_ratio(np.zeros(100, dtype=np.int8))
        assert result == 0.0


class TestAmplitudeCutoff:
    def test_typical(self):
        rng = _rng()
        amps = rng.normal(1.0, 0.3, 200)
        result = amplitude_cutoff(amps)
        assert 0 <= result <= 1

    def test_too_few(self):
        result = amplitude_cutoff(np.array([1.0, 2.0]))
        assert np.isnan(result)

    def test_peak_at_zero(self):
        # Force peak_idx == 0 by having most amplitudes near zero
        amps = np.concatenate([np.zeros(90), np.array([1.0] * 10)])
        result = amplitude_cutoff(amps)
        assert result == 0.5

    def test_empty_total(self):
        # Degenerate histogram where all bins empty (unlikely but safe)
        amps = np.zeros(20)
        result = amplitude_cutoff(amps)
        assert np.isfinite(result) or np.isnan(result)


class TestSNR:
    def test_typical(self):
        rng = _rng()
        waveforms = rng.normal(0, 0.1, (50, 30))
        waveforms[:, 15] += 2.0  # add peak
        result = snr(waveforms)
        assert result > 1

    def test_too_few(self):
        result = snr(np.array([[1, 2, 3]]))
        assert np.isnan(result)

    def test_zero_noise(self):
        waveforms = np.ones((5, 10))
        result = snr(waveforms)
        assert result == float("inf") or np.isfinite(result)


class TestNNHitRate:
    def test_typical(self):
        rng = _rng()
        cluster = rng.normal(0, 0.5, (20, 3))
        noise = rng.normal(5, 0.5, (20, 3))
        result = nn_hit_rate(cluster, noise, k=4)
        assert 0 <= result <= 1

    def test_too_small(self):
        cluster = _rng().normal(0, 1, (3, 2))
        noise = _rng().normal(3, 1, (10, 2))
        result = nn_hit_rate(cluster, noise, k=4)
        assert np.isnan(result)


class TestDriftMetric:
    def test_typical(self):
        rng = _rng()
        n = 100
        waveforms = rng.normal(0, 1, (n, 30))
        timestamps = np.arange(n, dtype=float)
        # Add drift
        waveforms[50:] *= 2
        result = drift_metric(waveforms, timestamps)
        assert result > 0

    def test_too_few(self):
        waveforms = _rng().normal(0, 1, (5, 10))
        timestamps = np.arange(5, dtype=float)
        result = drift_metric(waveforms, timestamps)
        assert np.isnan(result)

    def test_no_drift(self):
        waveforms = np.ones((20, 10))
        timestamps = np.arange(20, dtype=float)
        result = drift_metric(waveforms, timestamps)
        assert result == 0.0
