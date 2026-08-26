# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — JAX packed vector operation contracts

"""Verify vector AND/MAC values, shapes, dtypes, and dimensions."""

import numpy as np
import pytest

pytest.importorskip("jax")

from tests.jax_backend_support import jax_vec_and, jax_vec_mac, to_host


def test_jax_vec_and_validates_and_preserves_shape() -> None:
    a = np.array([[0b1111, 0b0011], [0b1010, 0b0101]], dtype=np.uint64)
    b = np.array([[0b1100, 0b0101], [0b0011, 0b1111]], dtype=np.uint64)
    result = to_host(jax_vec_and(a, b))
    assert result.shape == a.shape
    assert np.array_equal(result, np.bitwise_and(a, b))


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        (np.array([1, 3], dtype=np.uint64), np.array([1, 3], dtype=np.int64), "uint64"),
        (np.array([], dtype=np.uint64), np.array([], dtype=np.uint64), "non-empty"),
        (np.array([1, 3], dtype=np.uint64), np.array([[1, 3]], dtype=np.uint64), "shape"),
    ],
)
def test_jax_vec_and_rejects_invalid_contracts(a, b, match) -> None:
    with pytest.raises(ValueError, match=match):
        jax_vec_and(a, b)


def test_jax_vec_mac() -> None:
    weights = np.array([[[3], [1], [0]], [[255], [0], [7]]], dtype=np.uint64)
    inputs = np.array([[3], [3], [3]], dtype=np.uint64)
    out_np = to_host(jax_vec_mac(weights, inputs))
    assert out_np.shape == (2,)
    assert out_np[0] == 3
    assert out_np[1] == 4


@pytest.mark.parametrize(
    ("weights", "inputs", "match"),
    [
        (np.array([[3], [1]], dtype=np.uint64), np.array([[3], [3]], dtype=np.uint64), "3-D"),
        (np.array([[[3], [1]]], dtype=np.uint64), np.array([3, 3], dtype=np.uint64), "2-D"),
        (
            np.array([[[3], [1], [0]]], dtype=np.uint64),
            np.array([[3], [3]], dtype=np.uint64),
            "input dimension",
        ),
        (
            np.array([[[3], [1]]], dtype=np.float64),
            np.array([[3], [3]], dtype=np.uint64),
            "uint64",
        ),
    ],
)
def test_jax_vec_mac_rejects_invalid_contracts(weights, inputs, match) -> None:
    with pytest.raises(ValueError, match=match):
        jax_vec_mac(weights, inputs)
