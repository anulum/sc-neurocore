# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic attention validation tests

"""Configuration, shape, range, and finiteness validation for attention."""

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
def test_attention_invalid_configuration_raises(kwargs):  # type: ignore[no-untyped-def] # Preserved legacy test AST
    with pytest.raises(ValueError):
        StochasticAttention(**kwargs)


def test_forward_rejects_dimension_mismatch():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=4)
    Q = np.ones((2, 4))
    K = np.ones((3, 5))
    V = np.ones((3, 2))
    with pytest.raises(ValueError, match="K"):
        attn.forward(Q, K, V)


def test_forward_rejects_non_finite_inputs():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=2)
    Q = np.array([[np.nan, 0.0]])
    K = np.ones((1, 2))
    V = np.ones((1, 1))
    with pytest.raises(ValueError, match="finite"):
        attn.forward(Q, K, V)


def test_forward_bitstream_rejects_invalid_length():  # type: ignore[no-untyped-def] # Preserved legacy test AST
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
def test_forward_bitstream_rejects_out_of_range_probabilities(Q, K, V, message):  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=2)
    with pytest.raises(ValueError, match=message):
        attn.forward_bitstream(Q, K, V, length=8)


def test_forward_rejects_three_dimensional_inputs():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=4)
    cube = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        attn.forward(cube, cube, cube)


def test_forward_rejects_query_with_wrong_dim_k():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=4)
    Q = np.zeros((1, 3))  # three columns, expected four
    K = np.zeros((1, 4))
    V = np.zeros((1, 4))
    with pytest.raises(ValueError, match="dim_k=4 columns"):
        attn.forward(Q, K, V)


def test_forward_rejects_value_row_count_mismatch():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    attn = StochasticAttention(dim_k=4)
    Q = np.zeros((1, 4))
    K = np.zeros((2, 4))
    V = np.zeros((3, 4))  # rows must match K
    with pytest.raises(ValueError, match="same number of rows as K"):
        attn.forward(Q, K, V)
