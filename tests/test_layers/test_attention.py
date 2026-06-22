# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Python StochasticAttention forward pass

"""Tests for Python StochasticAttention forward pass."""

import numpy as np
import pytest

from sc_neurocore.layers.attention import StochasticAttention


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim_k": 0},
        {"dim_k": 4, "temperature": 0.0},
        {"dim_k": 4, "temperature": np.inf},
        {"dim_k": 4, "sc_mode": "bipolar"},
    ],
)
def test_attention_invalid_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        StochasticAttention(**kwargs)


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


def test_forward_rejects_dimension_mismatch():
    attn = StochasticAttention(dim_k=4)
    Q = np.ones((2, 4))
    K = np.ones((3, 5))
    V = np.ones((3, 2))
    with pytest.raises(ValueError, match="K"):
        attn.forward(Q, K, V)


def test_forward_rejects_non_finite_inputs():
    attn = StochasticAttention(dim_k=2)
    Q = np.array([[np.nan, 0.0]])
    K = np.ones((1, 2))
    V = np.ones((1, 1))
    with pytest.raises(ValueError, match="finite"):
        attn.forward(Q, K, V)


# -- forward_softmax tests --


def test_softmax_shape():
    attn = StochasticAttention(dim_k=8)
    Q = np.random.default_rng(0).uniform(0, 1, (4, 8))
    K = np.random.default_rng(1).uniform(0, 1, (6, 8))
    V = np.random.default_rng(2).uniform(0, 1, (6, 5))
    out = attn.forward_softmax(Q, K, V)
    assert out.shape == (4, 5)
    assert np.all(np.isfinite(out))


def test_softmax_single_key():
    attn = StochasticAttention(dim_k=4)
    Q = np.random.default_rng(0).uniform(0, 1, (3, 4))
    K = np.random.default_rng(1).uniform(0, 1, (1, 4))
    V = np.array([[7.0, 3.0]])
    out = attn.forward_softmax(Q, K, V)
    np.testing.assert_allclose(out, np.tile(V, (3, 1)), atol=1e-12)


def test_softmax_sharp_temperature():
    """Low temperature → winner-take-all: output ≈ V row of best-matching key."""
    attn = StochasticAttention(dim_k=4, temperature=0.01)
    Q = np.array([[1.0, 0.0, 0.0, 0.0]])
    K = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0]])
    out = attn.forward_softmax(Q, K, V)
    assert out[0, 0] > 9.0


def test_softmax_finite_on_large_scores():
    attn = StochasticAttention(dim_k=2, temperature=1.0)
    Q = np.full((1, 2), 1000.0)
    K = np.full((2, 2), 1000.0)
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = attn.forward_softmax(Q, K, V)
    assert np.all(np.isfinite(out))


def test_softmax_1d_inputs():
    attn = StochasticAttention(dim_k=4)
    q = np.array([0.5, 0.3, 0.2, 0.1])
    k = np.array([0.1, 0.2, 0.3, 0.4])
    v = np.array([1.0, 0.0])
    out = attn.forward_softmax(q, k, v)
    assert out.shape == (1, 2)
    assert np.all(np.isfinite(out))


def test_forward_bitstream_rejects_invalid_length():
    attn = StochasticAttention(dim_k=2)
    Q = np.ones((1, 2))
    K = np.ones((1, 2))
    V = np.ones((1, 1))
    with pytest.raises(ValueError, match="length"):
        attn.forward_bitstream(Q, K, V, length=0)


@pytest.mark.parametrize(
    ("Q", "K", "V", "message"),
    [
        (np.array([[1.1, 0.5]]), np.ones((1, 2)), np.ones((1, 1)), "Q"),
        (np.ones((1, 2)), np.array([[-0.1, 0.5]]), np.ones((1, 1)), "K"),
        (np.ones((1, 2)), np.ones((1, 2)), np.array([[1.1]]), "V"),
    ],
)
def test_forward_bitstream_rejects_out_of_range_probabilities(Q, K, V, message):
    attn = StochasticAttention(dim_k=2)
    with pytest.raises(ValueError, match=message):
        attn.forward_bitstream(Q, K, V, length=8)


def test_forward_rejects_three_dimensional_inputs():
    attn = StochasticAttention(dim_k=4)
    cube = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        attn.forward(cube, cube, cube)


def test_forward_rejects_query_with_wrong_dim_k():
    attn = StochasticAttention(dim_k=4)
    Q = np.zeros((1, 3))  # three columns, expected four
    K = np.zeros((1, 4))
    V = np.zeros((1, 4))
    with pytest.raises(ValueError, match="dim_k=4 columns"):
        attn.forward(Q, K, V)


def test_forward_rejects_value_row_count_mismatch():
    attn = StochasticAttention(dim_k=4)
    Q = np.zeros((1, 4))
    K = np.zeros((2, 4))
    V = np.zeros((3, 4))  # rows must match K
    with pytest.raises(ValueError, match="same number of rows as K"):
        attn.forward(Q, K, V)
