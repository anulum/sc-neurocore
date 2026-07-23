# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeSortingQuality from former test_spike_train_stats_extended.py

"""Focused suite: TestSpikeSortingQuality from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403

class TestSpikeSortingQuality:
    @pytest.fixture()
    def clusters(self):
        rng = np.random.default_rng(0)
        c = rng.normal(0, 0.5, (50, 3))
        n = rng.normal(5, 1.0, (100, 3))
        return c, n

    def test_isolation_distance(self, clusters):
        c, n = clusters
        iso = isolation_distance(c, n)
        assert iso > 0

    def test_l_ratio(self, clusters):
        c, n = clusters
        lr = l_ratio(c, n)
        assert np.isfinite(lr)

    def test_silhouette_score(self):
        rng = np.random.default_rng(1)
        data = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(5, 0.3, (30, 2))])
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        s = silhouette_score(data, labels)
        assert s > 0.5

    def test_d_prime(self, clusters):
        c, n = clusters
        dp = d_prime(c, n)
        assert dp > 0

    def test_isi_violation_rate(self, regular_train):
        vr = isi_violation_rate(regular_train)
        assert vr == 0.0

    def test_isi_violation_rate_with_violations(self):
        t = np.zeros(1000)
        t[[10, 11, 12, 50, 51, 200, 400]] = 1
        vr = isi_violation_rate(t, dt=0.001, refractory_ms=1.5)
        assert vr > 0

    def test_presence_ratio(self, poisson_train):
        pr = presence_ratio(poisson_train)
        assert 0.0 <= pr <= 1.0

    def test_amplitude_cutoff(self):
        rng = np.random.default_rng(4)
        amps = np.abs(rng.normal(100, 20, 500))
        ac = amplitude_cutoff(amps)
        assert 0.0 <= ac <= 1.0

    def test_snr(self):
        rng = np.random.default_rng(6)
        mean_wf = np.sin(np.linspace(0, 2 * np.pi, 40))
        waveforms = mean_wf[None, :] + rng.normal(0, 0.1, (100, 40))
        s = snr(waveforms)
        assert s > 1.0

    def test_nn_hit_rate(self, clusters):
        c, n = clusters
        hr = nn_hit_rate(c, n)
        assert 0.0 <= hr <= 1.0

    def test_drift_metric(self):
        rng = np.random.default_rng(2)
        wf = rng.normal(0, 1, (200, 30))
        wf[:100] *= 2.0
        ts = np.arange(200, dtype=np.float64)
        dm = drift_metric(wf, ts)
        assert dm > 0
