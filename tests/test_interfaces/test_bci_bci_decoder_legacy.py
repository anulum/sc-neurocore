# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBCIDecoderLegacy from former test_bci.py

"""Focused suite: TestBCIDecoderLegacy from former test_bci.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.interfaces.bci import (
    BCIDecoder,
)


class TestBCIDecoderLegacy:
    """Normalisation and bitstream encoding of the legacy BCI decoder."""

    def test_normalize_scales_to_unit(self) -> None:
        """Normalisation maps a signal onto the range zero to one."""
        decoder = BCIDecoder(channels=3)
        norm = decoder.normalize_signal(np.array([0.0, 2.0, 4.0]))
        assert np.isclose(norm.min(), 0.0)
        assert np.isclose(norm.max(), 1.0)

    def test_normalize_constant_returns_zero(self) -> None:
        """Normalising a constant signal gives zeros."""
        decoder = BCIDecoder(channels=2)
        norm = decoder.normalize_signal(np.array([5.0, 5.0]))
        assert np.allclose(norm, 0.0)

    def test_encode_bitstream_shape_2d(self) -> None:
        """A two-channel sample matrix encodes to one bitstream per channel."""
        decoder = BCIDecoder(channels=2)
        bits = decoder.encode_to_bitstream(np.array([[1.0, 2.0], [2.0, 3.0]]), length=16)
        assert bits.shape == (2, 16)

    def test_encode_bitstream_shape_1d(self) -> None:
        """A three-channel vector encodes to one bitstream per channel."""
        decoder = BCIDecoder(channels=3)
        bits = decoder.encode_to_bitstream(np.array([0.1, 0.2, 0.3]), length=8)
        assert bits.shape == (3, 8)

    def test_encode_bitstream_binary(self) -> None:
        """Encoded bitstreams hold only zeros and ones."""
        decoder = BCIDecoder(channels=2)
        bits = decoder.encode_to_bitstream(np.array([0.2, 0.8]), length=8)
        assert set(np.unique(bits).tolist()) <= {0, 1}

    def test_encode_length_mismatch_raises(self) -> None:
        """A signal with more channels than the decoder is refused."""
        decoder = BCIDecoder(channels=2)
        with pytest.raises(ValueError):
            decoder.encode_to_bitstream(np.array([0.1, 0.2, 0.3]), length=8)

    def test_negative_signal_normalizes(self) -> None:
        """A signal with negative values still normalises into zero to one."""
        decoder = BCIDecoder(channels=2)
        norm = decoder.normalize_signal(np.array([-1.0, 1.0]))
        assert np.all(norm >= 0.0)
        assert np.all(norm <= 1.0)

    def test_zero_signal_yields_zero_bits(self) -> None:
        """A zero signal encodes to all-zero bitstreams."""
        decoder = BCIDecoder(channels=2)
        bits = decoder.encode_to_bitstream(np.zeros(2), length=16)
        assert np.all(bits == 0)

    def test_deterministic_with_seed(self) -> None:
        """Two decoders with the same seed encode a signal alike."""
        d1 = BCIDecoder(channels=2, seed=42)
        d2 = BCIDecoder(channels=2, seed=42)
        signal = np.array([0.4, 0.6])
        bits_a = d1.encode_to_bitstream(signal, length=8)
        bits_b = d2.encode_to_bitstream(signal, length=8)
        assert np.array_equal(bits_a, bits_b)
