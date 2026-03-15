# SPDX-License-Identifier: AGPL-3.0-or-later
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
"""PyO3 round-trip tests for softmax attention."""

import numpy as np
import pytest

sc = pytest.importorskip("sc_neurocore_engine", exc_type=ImportError)



def test_forward_softmax_shape():
    attn = sc.StochasticAttention(dim_k=4)
    q = np.random.default_rng(1).random((2, 4))
    k = np.random.default_rng(2).random((3, 4))
    v = np.random.default_rng(3).random((3, 5))
    out = attn.forward_softmax(q, k, v)
    assert np.array(out).shape == (2, 5)



def test_forward_softmax_with_temperature():
    attn = sc.StochasticAttention(dim_k=4, temperature=0.01)
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    k = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    v = np.array([[10.0], [0.0]])
    out = np.array(attn.forward_softmax(q, k, v))
    # Sharp temperature: output ≈ first V row
    assert out[0, 0] > 9.0



def test_multihead_softmax_shape():
    attn = sc.StochasticAttention(dim_k=8)
    q = np.zeros((4, 8))
    k = np.zeros((4, 8))
    v = np.zeros((4, 8))
    out = np.array(attn.forward_multihead_softmax(q, k, v, n_heads=2))
    assert out.shape == (4, 8)



def test_softmax_finite_on_large_scores():
    attn = sc.StochasticAttention(dim_k=2, temperature=1.0)
    q = np.full((1, 2), 1000.0)
    k = np.full((2, 2), 1000.0)
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = np.array(attn.forward_softmax(q, k, v))
    assert np.all(np.isfinite(out))
