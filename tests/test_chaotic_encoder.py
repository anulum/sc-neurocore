# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for chaotic RNG → BitstreamEncoder integration

"""Tests for BitstreamEncoder mode='chaotic' (logistic map RNG)."""

import numpy as np
import pytest

from sc_neurocore.utils.bitstreams import BitstreamEncoder, bitstream_to_probability


class TestChaoticEncoder:
    def test_encode_returns_binary(self):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=256, mode="chaotic", seed=42)
        bits = enc.encode(0.5)
        assert bits.dtype == np.uint8
        assert set(np.unique(bits)).issubset({0, 1})

    def test_encode_shape(self):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=512, mode="chaotic", seed=42)
        bits = enc.encode(0.7)
        assert bits.shape == (512,)

    def test_probability_convergence(self):
        """Chaotic bitstream should converge to target probability."""
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=10000, mode="chaotic", seed=42)
        bits = enc.encode(0.6)
        p_hat = bitstream_to_probability(bits)
        np.testing.assert_allclose(p_hat, 0.6, atol=0.05)

    def test_decode_roundtrip(self):
        # Logistic map has arcsine distribution, so convergence is slower than Bernoulli
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=10000, mode="chaotic", seed=42)
        x = 0.5
        bits = enc.encode(x)
        x_rec = enc.decode(bits)
        np.testing.assert_allclose(x_rec, x, atol=0.1)

    def test_deterministic_same_seed(self):
        enc_a = BitstreamEncoder(x_min=0.0, x_max=1.0, length=100, mode="chaotic", seed=99)
        enc_b = BitstreamEncoder(x_min=0.0, x_max=1.0, length=100, mode="chaotic", seed=99)
        np.testing.assert_array_equal(enc_a.encode(0.5), enc_b.encode(0.5))

    def test_different_seeds_differ(self):
        enc_a = BitstreamEncoder(x_min=0.0, x_max=1.0, length=100, mode="chaotic", seed=1)
        enc_b = BitstreamEncoder(x_min=0.0, x_max=1.0, length=100, mode="chaotic", seed=2)
        assert not np.array_equal(enc_a.encode(0.5), enc_b.encode(0.5))

    def test_zero_encodes_all_zeros(self):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=200, mode="chaotic", seed=42)
        bits = enc.encode(0.0)
        assert np.all(bits == 0)

    def test_one_encodes_all_ones(self):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=200, mode="chaotic", seed=42)
        bits = enc.encode(1.0)
        assert np.all(bits == 1)
