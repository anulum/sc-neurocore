# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCodecRegistry from former test_codec_registry.py

"""Focused suite: TestCodecRegistry from former test_codec_registry.py."""

from __future__ import annotations

from tests.codec_registry_support import *  # noqa: F403


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
