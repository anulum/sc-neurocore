# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamRateDecoder from former test_encoding.py

"""Focused suite: TestBitstreamRateDecoder from former test_encoding.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.bioware.bioware import (
    decode_bitstream_rate,
)


class TestBitstreamRateDecoder:
    """Decoding firing rates from bitstreams."""

    def test_full_density(self) -> None:
        """A full bitstream decodes to the clock rate."""
        bs = {0: np.ones(256, dtype=np.uint8)}
        rates = decode_bitstream_rate(bs, sc_clock_hz=1e6)
        assert rates[0] == 1e6

    def test_half_density(self) -> None:
        """A half bitstream decodes to half the clock rate."""
        bs_data = np.zeros(256, dtype=np.uint8)
        bs_data[:128] = 1
        rates = decode_bitstream_rate({0: bs_data}, sc_clock_hz=1e6)
        assert rates[0] == pytest.approx(500000.0)

    def test_empty_bitstream(self) -> None:
        """An empty bitstream decodes to rate 0."""
        rates = decode_bitstream_rate({0: np.array([], dtype=np.uint8)})
        assert rates[0] == 0.0
