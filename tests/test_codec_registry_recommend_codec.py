# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRecommendCodec from former test_codec_registry.py

"""Focused suite: TestRecommendCodec from former test_codec_registry.py."""

from __future__ import annotations

from tests.codec_registry_support import *  # noqa: F403

class TestRecommendCodec:
    def test_neuromorphic_gets_aer(self):
        assert recommend_codec(64, 5.0, neuromorphic=True) == "aer"

    def test_low_latency_gets_streaming(self):
        assert recommend_codec(1024, 2.0, latency_ms=0.5) == "streaming"

    def test_correlated_gets_delta(self):
        assert recommend_codec(384, 3.0, correlated=True) == "delta"

    def test_high_channel_gets_predictive(self):
        assert recommend_codec(1024, 2.0) == "predictive"

    def test_small_gets_isi(self):
        assert recommend_codec(8, 5.0) == "isi"

    def test_recommendation_is_valid_codec(self):
        """Every recommendation must be a registered codec."""
        for n in [1, 8, 64, 384, 1024]:
            for rate in [0.5, 5.0, 50.0]:
                for lat in [0.5, 1.0, 10.0]:
                    for corr in [True, False]:
                        for neuro in [True, False]:
                            name = recommend_codec(n, rate, lat, corr, neuro)
                            assert name in CODEC_REGISTRY
