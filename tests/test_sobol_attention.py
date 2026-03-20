# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Sobol SC attention (Tier 1.8)

"""Tests for bitstream-level attention with Sobol variance reduction."""

import numpy as np

from sc_neurocore.layers.attention import StochasticAttention


class TestBitstreamAttention:
    def test_output_shape(self):
        attn = StochasticAttention(dim_k=4)
        Q = np.random.uniform(0, 1, (3, 4))
        K = np.random.uniform(0, 1, (5, 4))
        V = np.random.uniform(0, 1, (5, 2))
        out = attn.forward_bitstream(Q, K, V, length=256)
        assert out.shape == (3, 2)

    def test_output_bounded(self):
        attn = StochasticAttention(dim_k=4)
        Q = np.random.uniform(0, 1, (2, 4))
        K = np.random.uniform(0, 1, (3, 4))
        V = np.random.uniform(0, 1, (3, 2))
        out = attn.forward_bitstream(Q, K, V, length=512)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_matches_float_attention(self):
        """Bitstream attention should approximate float attention."""
        attn = StochasticAttention(dim_k=4)
        np.random.seed(42)
        Q = np.random.uniform(0.2, 0.8, (2, 4))
        K = np.random.uniform(0.2, 0.8, (3, 4))
        V = np.random.uniform(0.2, 0.8, (3, 2))
        float_out = attn.forward(Q, K, V)
        bs_out = attn.forward_bitstream(Q, K, V, length=4096)
        np.testing.assert_allclose(bs_out, float_out, atol=0.15)

    def test_sobol_produces_valid_output(self):
        """Sobol bitstream attention should produce bounded valid output."""
        attn = StochasticAttention(dim_k=4)
        np.random.seed(42)
        Q = np.random.uniform(0.2, 0.8, (2, 4))
        K = np.random.uniform(0.2, 0.8, (4, 4))
        V = np.random.uniform(0.2, 0.8, (4, 3))
        out = attn.forward_bitstream(Q, K, V, length=1024, use_sobol=True)
        assert out.shape == (2, 3)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)
        assert np.all(np.isfinite(out))

    def test_sobol_flag_works(self):
        """use_sobol=True should produce valid output."""
        attn = StochasticAttention(dim_k=2)
        Q = np.array([[0.3, 0.7]])
        K = np.array([[0.5, 0.5]])
        V = np.array([[0.6, 0.4]])
        out = attn.forward_bitstream(Q, K, V, length=512, use_sobol=True)
        assert out.shape == (1, 2)
        assert np.all(np.isfinite(out))
