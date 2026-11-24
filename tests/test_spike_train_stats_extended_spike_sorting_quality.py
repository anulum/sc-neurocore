# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeSortingQuality from former test_spike_train_stats_extended.py

"""Focused suite: TestSpikeSortingQuality from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import amplitude_cutoff
from sc_neurocore.analysis.spike_stats import d_prime
from sc_neurocore.analysis.spike_stats import drift_metric
from sc_neurocore.analysis.spike_stats import isi_violation_rate
from sc_neurocore.analysis.spike_stats import isolation_distance
from sc_neurocore.analysis.spike_stats import l_ratio
from sc_neurocore.analysis.spike_stats import nn_hit_rate
from sc_neurocore.analysis.spike_stats import presence_ratio
from sc_neurocore.analysis.spike_stats import silhouette_score
from sc_neurocore.analysis.spike_stats import snr
import numpy as np
import pytest
from tests.spike_train_stats_extended_support import poisson_train as poisson_train
from tests.spike_train_stats_extended_support import regular_train as regular_train


class TestSpikeSortingQuality:
    """Spike-sorting quality metrics on separated clusters and spike trains."""

    @pytest.fixture()
    def clusters(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return 50 cluster points around 0 and 100 noise points around 5 in three features."""
        rng = np.random.default_rng(0)
        c = rng.normal(0, 0.5, (50, 3))
        n = rng.normal(5, 1.0, (100, 3))
        return c, n

    def test_isolation_distance(
        self, clusters: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """Separated clusters give a positive isolation distance."""
        c, n = clusters
        iso = isolation_distance(c, n)
        assert iso > 0

    def test_l_ratio(self, clusters: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]) -> None:
        """Separated clusters give a finite L-ratio."""
        c, n = clusters
        lr = l_ratio(c, n)
        assert np.isfinite(lr)

    def test_silhouette_score(self) -> None:
        """Two well-separated classes give a silhouette score above 0.5."""
        rng = np.random.default_rng(1)
        data = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(5, 0.3, (30, 2))])
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        s = silhouette_score(data, labels)
        assert s > 0.5

    def test_d_prime(self, clusters: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]) -> None:
        """Separated clusters give a positive d-prime."""
        c, n = clusters
        dp = d_prime(c, n)
        assert dp > 0

    def test_isi_violation_rate(self, regular_train: np.ndarray[Any, Any]) -> None:
        """A regular train has no ISI violations."""
        vr = isi_violation_rate(regular_train)
        assert vr == 0.0

    def test_isi_violation_rate_with_violations(self) -> None:
        """Spikes 1 ms apart give a positive violation rate for a 1.5 ms refractory period."""
        t = np.zeros(1000)
        t[[10, 11, 12, 50, 51, 200, 400]] = 1
        vr = isi_violation_rate(t, dt=0.001, refractory_ms=1.5)
        assert vr > 0

    def test_presence_ratio(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The presence ratio of a Poisson train lies inside [0, 1]."""
        pr = presence_ratio(poisson_train)
        assert 0.0 <= pr <= 1.0

    def test_amplitude_cutoff(self) -> None:
        """The amplitude cutoff of positive normal amplitudes lies inside [0, 1]."""
        rng = np.random.default_rng(4)
        amps = np.abs(rng.normal(100, 20, 500))
        ac = amplitude_cutoff(amps)
        assert 0.0 <= ac <= 1.0

    def test_snr(self) -> None:
        """A sinusoidal mean waveform under noise of sd 0.1 gives an SNR above one."""
        rng = np.random.default_rng(6)
        mean_wf = np.sin(np.linspace(0, 2 * np.pi, 40))
        waveforms = mean_wf[None, :] + rng.normal(0, 0.1, (100, 40))
        s = snr(waveforms)
        assert s > 1.0

    def test_nn_hit_rate(self, clusters: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]) -> None:
        """The nearest-neighbour hit rate of separated clusters lies inside [0, 1]."""
        c, n = clusters
        hr = nn_hit_rate(c, n)
        assert 0.0 <= hr <= 1.0

    def test_drift_metric(self) -> None:
        """Waveforms doubled in amplitude in their first half give a positive drift."""
        rng = np.random.default_rng(2)
        wf = rng.normal(0, 1, (200, 30))
        wf[:100] *= 2.0
        ts = np.arange(200, dtype=np.float64)
        dm = drift_metric(wf, ts)
        assert dm > 0
