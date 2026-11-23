# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERSpikeCodecEdgeCases from former test_aer_spike_codec.py

"""Focused suite: TestAERSpikeCodecEdgeCases from former test_aer_spike_codec.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.spike_codec.aer_codec import (
    AERCompressionResult,
    AERSpikeCodec,
)


class TestAERSpikeCodecEdgeCases:
    """Edge cases of the AER spike codec."""

    def test_invalid_magic_raises(self) -> None:
        """A stream with a bad header magic is rejected."""
        codec = AERSpikeCodec()
        with pytest.raises(ValueError, match="Invalid header magic"):
            codec.decompress(b"XXXX" + b"\x00" * 100)

    def test_result_type(self) -> None:
        """Compression returns an AER result record."""
        spikes = np.zeros((10, 4), dtype=np.int8)
        _, result = AERSpikeCodec().compress(spikes)
        assert isinstance(result, AERCompressionResult)
        assert result.codec_type == "aer"

    def test_single_channel(self) -> None:
        """A single channel round-trips unchanged."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 1)) < 0.05).astype(np.int8)
        codec = AERSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
