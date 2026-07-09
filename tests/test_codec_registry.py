# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.spike_codec.registry

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.spike_codec.registry import (
    get_codec,
    list_codecs,
    recommend_codec,
    CODEC_REGISTRY,
)


class TestCodecRegistry:
    def test_list_codecs(self):
        codecs = list_codecs()
        assert "isi" in codecs
        assert "predictive" in codecs
        assert "delta" in codecs
        assert "streaming" in codecs
        assert "aer" in codecs
        assert len(codecs) == 5

    def test_get_all_codecs(self):
        for name in list_codecs():
            codec = get_codec(name)
            assert hasattr(codec, "compress")
            assert hasattr(codec, "decompress")

    def test_get_codec_with_kwargs(self):
        c = get_codec("predictive", alpha=0.01)
        assert c.alpha == 0.01
        c = get_codec("delta", group_size=16)
        assert c.group_size == 16
        c = get_codec("streaming", window_size=10)
        assert c.window_size == 10

    def test_unknown_codec_raises(self):
        with pytest.raises(ValueError, match="Unknown codec"):
            get_codec("nonexistent")

    def test_all_codecs_roundtrip(self):
        """Every codec must roundtrip the same data."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 16)) < 0.05).astype(np.int8)

        for name in list_codecs():
            codec = get_codec(name)
            data, result = codec.compress(spikes)
            if name in ("streaming", "aer"):
                recovered = codec.decompress(data)
            else:
                recovered = codec.decompress(data, 100, 16)
            np.testing.assert_array_equal(recovered, spikes, err_msg=f"Roundtrip failed for {name}")


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
