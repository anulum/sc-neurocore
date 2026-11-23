# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStreamingSpikeCodecEdgeCases from former test_streaming_spike_codec.py

"""Focused suite: TestStreamingSpikeCodecEdgeCases from former test_streaming_spike_codec.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.spike_codec.streaming_codec import (
    StreamingCompressionResult,
    StreamingSpikeCodec,
)


class TestStreamingSpikeCodecEdgeCases:
    """Edge cases of the streaming codec."""

    def test_invalid_magic_raises(self) -> None:
        """A stream with a wrong header magic is rejected."""
        codec = StreamingSpikeCodec()
        with pytest.raises(ValueError, match="Invalid header magic"):
            codec.decompress(b"XXXX" + b"\x00" * 100)

    def test_result_type(self) -> None:
        """Compression returns a streaming compression result."""
        spikes = np.zeros((20, 4), dtype=np.int8)
        _, result = StreamingSpikeCodec().compress(spikes)
        assert isinstance(result, StreamingCompressionResult)
