# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.encoding (encoding zoo + optimizer)

from __future__ import annotations

import numpy as np

from sc_neurocore.encoding import (
    rate_encode,
    latency_encode,
    delta_encode,
    phase_encode,
    burst_encode,
    rank_order_encode,
    sigma_delta_encode,
    EncodingOptimizer,
)


class TestRateEncode:
    def test_shape(self):
        s = rate_encode(np.array([0.5, 0.3, 0.8]), T=20)
        assert s.shape == (20, 3)
        assert s.dtype == np.int8

    def test_rate_correlation(self):
        s = rate_encode(np.array([0.1, 0.9]), T=1000, seed=42)
        assert s[:, 1].mean() > s[:, 0].mean()


class TestLatencyEncode:
    def test_shape(self):
        s = latency_encode(np.array([0.5, 0.8, 0.2]), T=10)
        assert s.shape == (10, 3)

    def test_higher_value_fires_earlier(self):
        s = latency_encode(np.array([0.2, 0.9]), T=20)
        t_low = np.argmax(s[:, 0])
        t_high = np.argmax(s[:, 1])
        assert t_high <= t_low


class TestDeltaEncode:
    def test_1d(self):
        signal = np.array([0.0, 0.0, 0.5, 0.5, 1.0])
        s = delta_encode(signal, threshold=0.3)
        assert s.shape == (5, 1)
        assert s[2, 0] == 1  # change 0→0.5

    def test_2d(self):
        signal = np.random.rand(20, 4)
        s = delta_encode(signal, threshold=0.1)
        assert s.shape == (20, 4)


class TestPhaseEncode:
    def test_shape(self):
        s = phase_encode(np.array([0.5, 0.3]), T=16, n_phases=4)
        assert s.shape == (16, 2)

    def test_periodic(self):
        s = phase_encode(np.array([0.5]), T=24, n_phases=8)
        assert s[:, 0].sum() >= 3  # fires every 8 steps at phase 4


class TestBurstEncode:
    def test_shape(self):
        s = burst_encode(np.array([0.2, 0.8]), T=10, max_burst=5)
        assert s.shape == (10, 2)

    def test_higher_value_longer_burst(self):
        s = burst_encode(np.array([0.2, 1.0]), T=10, max_burst=5)
        assert s[:, 1].sum() >= s[:, 0].sum()


class TestRankOrderEncode:
    def test_shape(self):
        s = rank_order_encode(np.array([0.3, 0.9, 0.1, 0.7]), T=10)
        assert s.shape == (10, 4)

    def test_order(self):
        s = rank_order_encode(np.array([0.1, 0.9, 0.5]), T=10)
        t_high = np.argmax(s[:, 1])
        t_low = np.argmax(s[:, 0])
        assert t_high <= t_low


class TestSigmaDeltaEncode:
    def test_1d(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        s = sigma_delta_encode(signal, threshold=0.2)
        assert s.shape == (100, 1)
        assert s.sum() > 0

    def test_2d(self):
        signal = np.random.rand(50, 3)
        s = sigma_delta_encode(signal, threshold=0.1)
        assert s.shape == (50, 3)


class TestEncodingOptimizer:
    def test_profile(self):
        opt = EncodingOptimizer(T=16)
        data = np.random.rand(100)
        stats = opt.profile(data)
        assert "mean" in stats
        assert "std" in stats
        assert "sparsity" in stats
        assert "dynamic_range" in stats

    def test_profile_2d(self):
        opt = EncodingOptimizer(T=16)
        data = np.random.rand(50, 10)
        stats = opt.profile(data)
        assert "temporal_autocorrelation" in stats

    def test_recommend(self):
        opt = EncodingOptimizer(T=16)
        data = np.random.rand(32)
        recs = opt.recommend(data)
        assert len(recs) >= 5
        assert recs[0].score >= recs[-1].score
        for r in recs:
            assert 0 <= r.score <= 1
            assert r.encoding != ""
            assert r.reason != ""

    def test_recommend_sparse_data(self):
        opt = EncodingOptimizer(T=16)
        data = np.zeros(100)
        data[:5] = 1.0
        recs = opt.recommend(data)
        assert len(recs) >= 5

    def test_unnormalized_data(self):
        opt = EncodingOptimizer(T=16)
        data = np.random.rand(32) * 100 - 50
        recs = opt.recommend(data)
        assert len(recs) >= 5
