# SPDX-License-Identifier: AGPL-3.0-or-later
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Python StochasticAttention forward pass."""

import numpy as np

from sc_neurocore.layers.attention import StochasticAttention


def test_forward_shape():
    attn = StochasticAttention(dim_k=8)
    Q = np.random.default_rng(0).uniform(0, 1, (4, 8))
    K = np.random.default_rng(1).uniform(0, 1, (6, 8))
    V = np.random.default_rng(2).uniform(0, 1, (6, 5))
    out = attn.forward(Q, K, V)
    assert out.shape == (4, 5)
    assert np.all(np.isfinite(out))


def test_forward_1d_inputs():
    attn = StochasticAttention(dim_k=4)
    q = np.array([0.5, 0.3, 0.2, 0.1])
    k = np.array([0.1, 0.2, 0.3, 0.4])
    v = np.array([1.0, 0.0])
    out = attn.forward(q, k, v)
    assert out.shape == (1, 2)
    assert np.all(np.isfinite(out))


def test_forward_zero_queries():
    attn = StochasticAttention(dim_k=4)
    Q = np.zeros((3, 4))
    K = np.random.default_rng(0).uniform(0, 1, (5, 4))
    V = np.random.default_rng(1).uniform(0, 1, (5, 2))
    out = attn.forward(Q, K, V)
    assert out.shape == (3, 2)
    assert np.all(np.isfinite(out))


def test_forward_single_key():
    attn = StochasticAttention(dim_k=4)
    Q = np.random.default_rng(0).uniform(0, 1, (3, 4))
    K = np.random.default_rng(1).uniform(0, 1, (1, 4))
    V = np.array([[7.0, 3.0]])
    out = attn.forward(Q, K, V)
    # With a single key, all queries must attend to it -> output = V
    np.testing.assert_allclose(out, np.tile(V, (3, 1)), atol=1e-12)
