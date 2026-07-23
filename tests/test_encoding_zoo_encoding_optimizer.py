# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEncodingOptimizer from former test_encoding_zoo.py

"""Focused suite: TestEncodingOptimizer from former test_encoding_zoo.py."""

from __future__ import annotations

from tests.encoding_zoo_support import *  # noqa: F403

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
