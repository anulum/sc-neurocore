# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStreamingSpikeCodecEdgeCases from former test_streaming_spike_codec.py

"""Focused suite: TestStreamingSpikeCodecEdgeCases from former test_streaming_spike_codec.py."""

from __future__ import annotations

from tests.streaming_spike_codec_support import *  # noqa: F403

class TestStreamingSpikeCodecEdgeCases:
    def test_invalid_magic_raises(self):
        codec = StreamingSpikeCodec()
        with pytest.raises(ValueError, match="Invalid header magic"):
            codec.decompress(b"XXXX" + b"\x00" * 100)

    def test_result_type(self):
        spikes = np.zeros((20, 4), dtype=np.int8)
        _, result = StreamingSpikeCodec().compress(spikes)
        assert isinstance(result, StreamingCompressionResult)
