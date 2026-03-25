# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for BCI encoder (legacy BCIDecoder + new BCIEncoder)

"""Tests for BCI signal encoding: legacy API + new deterministic API."""

import numpy as np
import pytest

from sc_neurocore.interfaces.bci import BCIDecoder, BCIEncoder


# --- Legacy BCIDecoder API (backward compatibility) ---


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


# --- New BCIEncoder API ---


class TestBCIEncoder:
    def test_encode_shape(self):
        enc = BCIEncoder(n_channels=4, seed=42)
        spikes = enc.encode(np.array([0.1, 0.5, 0.8, 0.3]), T=20)
        assert spikes.shape == (20, 4)
        assert spikes.dtype == np.int8

    def test_encode_binary_output(self):
        enc = BCIEncoder(n_channels=8, seed=42)
        spikes = enc.encode(np.random.randn(8), T=50)
        assert set(np.unique(spikes).tolist()) <= {0, 1}

    def test_encode_deterministic(self):
        """Same seed → same output, always."""
        enc1 = BCIEncoder(n_channels=4, seed=99)
        enc2 = BCIEncoder(n_channels=4, seed=99)
        signal = np.array([0.2, 0.4, 0.6, 0.8])
        assert np.array_equal(enc1.encode(signal, T=30), enc2.encode(signal, T=30))

    def test_encode_2d_input(self):
        """Multi-sample input averaged per channel."""
        enc = BCIEncoder(n_channels=3, seed=42)
        signal = np.random.randn(3, 100)
        spikes = enc.encode(signal, T=20)
        assert spikes.shape == (20, 3)

    def test_encode_stream(self):
        enc = BCIEncoder(n_channels=4, sampling_rate=20000, window_ms=1.0, seed=42)
        signal = np.random.RandomState(42).randn(4, 1000)
        stream = enc.encode_stream(signal)
        assert stream.shape[1] == 4
        assert stream.shape[0] > 0
        assert stream.dtype == np.int8

    def test_encode_stream_empty(self):
        enc = BCIEncoder(n_channels=2, sampling_rate=20000, window_ms=1.0)
        signal = np.random.randn(2, 5)  # too short for one window
        stream = enc.encode_stream(signal)
        assert stream.shape[1] == 2

    def test_normalize_flat_signal(self):
        result = BCIEncoder._normalize(np.array([5.0, 5.0, 5.0]))
        assert np.allclose(result, 0.5)
