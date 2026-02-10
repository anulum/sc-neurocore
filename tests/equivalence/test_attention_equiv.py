"""Equivalence: v2 StochasticAttention vs v3."""

import numpy as np
import pytest

from sc_neurocore.layers.attention import StochasticAttention as V2Attn
from sc_neurocore_engine.attention import StochasticAttention as V3Attn


class TestAttentionEquivalence:
    @pytest.mark.parametrize(
        "n,m,dk,dv",
        [
            (1, 1, 4, 4),
            (5, 10, 8, 16),
            (10, 10, 16, 32),
            (20, 50, 32, 64),
        ],
    )
    def test_forward_matches_v2(self, n, m, dk, dv):
        rng = np.random.RandomState(42)
        Q = rng.uniform(0, 1, (n, dk))
        K = rng.uniform(0, 1, (m, dk))
        V = rng.uniform(0, 1, (m, dv))

        v2 = V2Attn(dim_k=dk)
        v3 = V3Attn(dim_k=dk)

        v2_out = v2.forward(Q, K, V)
        v3_out = v3.forward(Q, K, V)

        np.testing.assert_allclose(
            v2_out,
            v3_out,
            atol=1e-12,
            err_msg="Attention output mismatch (rate mode)",
        )

    def test_1d_input_expansion(self):
        """v2 handles 1-D inputs by expanding to 2-D. v3 must too."""
        rng = np.random.RandomState(42)
        q = rng.uniform(0, 1, 8)
        k = rng.uniform(0, 1, 8)
        v = rng.uniform(0, 1, 4)

        v2 = V2Attn(dim_k=8)
        v3 = V3Attn(dim_k=8)

        v2_out = v2.forward(q, k, v)
        v3_out = v3.forward(q, k, v)

        np.testing.assert_allclose(v2_out, v3_out, atol=1e-12)

    def test_zero_scores_handling(self):
        """When Q and K are zero, row_sums = 0. Must not crash."""
        Q = np.zeros((3, 4))
        K = np.zeros((5, 4))
        V = np.random.RandomState(42).uniform(0, 1, (5, 8))

        v2 = V2Attn(dim_k=4)
        v3 = V3Attn(dim_k=4)

        v2_out = v2.forward(Q, K, V)
        v3_out = v3.forward(Q, K, V)

        np.testing.assert_allclose(v2_out, v3_out, atol=1e-12)
