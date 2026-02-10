"""Tests for multi-head attention (new v3 capability)."""

import numpy as np

from sc_neurocore_engine.attention import StochasticAttention


class TestMultiHeadAttention:
    def test_single_head_matches_forward(self):
        """With n_heads=1, forward_multihead == forward."""
        rng = np.random.RandomState(42)
        Q = rng.uniform(0, 1, (5, 8))
        K = rng.uniform(0, 1, (10, 8))
        V = rng.uniform(0, 1, (10, 4))

        attn = StochasticAttention(dim_k=8)
        out_single = attn.forward(Q, K, V)
        out_multi = attn.forward_multihead(Q, K, V, n_heads=1)

        np.testing.assert_allclose(out_single, out_multi, atol=1e-12)

    def test_multi_head_output_shape(self):
        """Multi-head output should have same total columns as V."""
        rng = np.random.RandomState(42)
        n_heads = 4
        Q = rng.uniform(0, 1, (5, 32))
        K = rng.uniform(0, 1, (10, 32))
        V = rng.uniform(0, 1, (10, 16))

        attn = StochasticAttention(dim_k=32)
        out = attn.forward_multihead(Q, K, V, n_heads=n_heads)

        assert out.shape == (5, 16)

    def test_multi_head_not_equal_to_single(self):
        """Multi-head should differ from treating all columns as one head."""
        rng = np.random.RandomState(42)
        Q = rng.uniform(0, 1, (5, 16))
        K = rng.uniform(0, 1, (10, 16))
        V = rng.uniform(0, 1, (10, 8))

        attn = StochasticAttention(dim_k=16)
        out_1head = attn.forward_multihead(Q, K, V, n_heads=1)
        out_2head = attn.forward_multihead(Q, K, V, n_heads=2)

        assert not np.allclose(out_1head, out_2head)
