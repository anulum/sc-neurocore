# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBCIDecoderLegacy from former test_bci.py

"""Focused suite: TestBCIDecoderLegacy from former test_bci.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from bci_support import *  # noqa: F403

class TestBCIDecoderLegacy:
    def test_normalize_scales_to_unit(self):
        decoder = BCIDecoder(channels=3)
        norm = decoder.normalize_signal(np.array([0.0, 2.0, 4.0]))
        assert np.isclose(norm.min(), 0.0)
        assert np.isclose(norm.max(), 1.0)

    def test_normalize_constant_returns_zero(self):
        decoder = BCIDecoder(channels=2)
        norm = decoder.normalize_signal(np.array([5.0, 5.0]))
        assert np.allclose(norm, 0.0)

    def test_encode_bitstream_shape_2d(self):
        decoder = BCIDecoder(channels=2)
        bits = decoder.encode_to_bitstream(np.array([[1.0, 2.0], [2.0, 3.0]]), length=16)
        assert bits.shape == (2, 16)

    def test_encode_bitstream_shape_1d(self):
        decoder = BCIDecoder(channels=3)
        bits = decoder.encode_to_bitstream(np.array([0.1, 0.2, 0.3]), length=8)
        assert bits.shape == (3, 8)

    def test_encode_bitstream_binary(self):
        decoder = BCIDecoder(channels=2)
        bits = decoder.encode_to_bitstream(np.array([0.2, 0.8]), length=8)
        assert set(np.unique(bits).tolist()) <= {0, 1}

    def test_encode_length_mismatch_raises(self):
        decoder = BCIDecoder(channels=2)
        with pytest.raises(ValueError):
            decoder.encode_to_bitstream(np.array([0.1, 0.2, 0.3]), length=8)

    def test_negative_signal_normalizes(self):
        decoder = BCIDecoder(channels=2)
        norm = decoder.normalize_signal(np.array([-1.0, 1.0]))
        assert np.all(norm >= 0.0)
        assert np.all(norm <= 1.0)

    def test_zero_signal_yields_zero_bits(self):
        decoder = BCIDecoder(channels=2)
        bits = decoder.encode_to_bitstream(np.zeros(2), length=16)
        assert np.all(bits == 0)

    def test_deterministic_with_seed(self):
        d1 = BCIDecoder(channels=2, seed=42)
        d2 = BCIDecoder(channels=2, seed=42)
        signal = np.array([0.4, 0.6])
        bits_a = d1.encode_to_bitstream(signal, length=8)
        bits_b = d2.encode_to_bitstream(signal, length=8)
        assert np.array_equal(bits_a, bits_b)
