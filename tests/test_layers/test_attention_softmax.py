# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic attention softmax tests

"""Shape, key-count, temperature, stability, and vector softmax contracts."""

import numpy as np

from sc_neurocore.layers.attention import StochasticAttention


def test_softmax_shape():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=8)
    Q = np.random.default_rng(0).uniform(0, 1, (4, 8))
    K = np.random.default_rng(1).uniform(0, 1, (6, 8))
    V = np.random.default_rng(2).uniform(0, 1, (6, 5))
    out = attn.forward_softmax(Q, K, V)
    assert out.shape == (4, 5)
    assert np.all(np.isfinite(out))


def test_softmax_single_key():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=4)
    Q = np.random.default_rng(0).uniform(0, 1, (3, 4))
    K = np.random.default_rng(1).uniform(0, 1, (1, 4))
    V = np.array([[7.0, 3.0]])
    out = attn.forward_softmax(Q, K, V)
    np.testing.assert_allclose(out, np.tile(V, (3, 1)), atol=1e-12)


def test_softmax_sharp_temperature():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Low temperature → winner-take-all: output ≈ V row of best-matching key."""
    attn = StochasticAttention(dim_k=4, temperature=0.01)
    Q = np.array([[1.0, 0.0, 0.0, 0.0]])
    K = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0]])
    out = attn.forward_softmax(Q, K, V)
    assert out[0, 0] > 9.0


def test_softmax_finite_on_large_scores():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=2, temperature=1.0)
    Q = np.full((1, 2), 1000.0)
    K = np.full((2, 2), 1000.0)
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = attn.forward_softmax(Q, K, V)
    assert np.all(np.isfinite(out))


def test_softmax_1d_inputs():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=4)
    q = np.array([0.5, 0.3, 0.2, 0.1])
    k = np.array([0.1, 0.2, 0.3, 0.4])
    v = np.array([1.0, 0.0])
    out = attn.forward_softmax(q, k, v)
    assert out.shape == (1, 2)
    assert np.all(np.isfinite(out))
