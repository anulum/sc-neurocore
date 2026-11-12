# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bit-input contract of the bipolar decode and multiply primitives

"""Bind the bit-input contract of ``bipolar_decode`` and ``bipolar_multiply``.

Bits are checked in the representation the caller gave, before conversion to
one byte per bit, so no value becomes 0 or 1 through a cast. Any shape is
admitted.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

from sc_neurocore.core import bipolar_decode, bipolar_multiply

_NON_BINARY_INPUTS: dict[str, ArrayLike] = {
    "fractional floats": np.array([0.9, 1.9]),
    "wrapping unsigned integers": np.array([256, 257], dtype=np.uint16),
    "negative integer": np.array([-1, 0], dtype=np.int64),
    "integer beyond one byte": np.array([0, 2**40], dtype=np.int64),
    "not a number": np.array([0.0, np.nan]),
    "infinity": np.array([1.0, np.inf]),
    "values near one": np.array([1.0 - 1e-12, 1.0]),
}

_INADMISSIBLE_TYPES: dict[str, ArrayLike] = {
    "text": ["0", "1"],
    "bytes": [b"0", b"1"],
    "objects": np.array([0, 1, None], dtype=object),
    "complex": np.array([0j, 1 + 0j]),
    "integers too large for a machine word": [0, 2**70],
}

_REFERENCE_A = np.array([[1, 0, 1, 1], [0, 0, 1, 0]], dtype=np.uint8)
_REFERENCE_B = np.array([[1, 1, 0, 1], [0, 1, 0, 0]], dtype=np.uint8)

_BINARY_REPRESENTATIONS = {
    "nested python integers": (_REFERENCE_A.tolist(), _REFERENCE_B.tolist()),
    "nested python booleans": (
        _REFERENCE_A.astype(bool).tolist(),
        _REFERENCE_B.astype(bool).tolist(),
    ),
    "boolean array": (_REFERENCE_A.astype(np.bool_), _REFERENCE_B.astype(np.bool_)),
    "signed integers": (_REFERENCE_A.astype(np.int64), _REFERENCE_B.astype(np.int64)),
    "exact floats": (_REFERENCE_A.astype(np.float64), _REFERENCE_B.astype(np.float64)),
}


@pytest.mark.parametrize("case", sorted(_NON_BINARY_INPUTS))
def test_decode_refuses_values_other_than_zero_and_one_before_conversion(case: str) -> None:
    """A value that only becomes a bit through a cast to one byte is refused."""
    with pytest.raises(ValueError, match="bits must contain only 0/1 bits"):
        bipolar_decode(_NON_BINARY_INPUTS[case])


@pytest.mark.parametrize("case", sorted(_NON_BINARY_INPUTS))
def test_multiply_refuses_values_other_than_zero_and_one_before_conversion(case: str) -> None:
    """Either operand is refused by its own name."""
    valid = np.array([0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="a must contain only 0/1 bits"):
        bipolar_multiply(_NON_BINARY_INPUTS[case], valid)
    with pytest.raises(ValueError, match="b must contain only 0/1 bits"):
        bipolar_multiply(valid, _NON_BINARY_INPUTS[case])


@pytest.mark.parametrize("case", sorted(_INADMISSIBLE_TYPES))
def test_inadmissible_element_types_are_refused(case: str) -> None:
    """Text, bytes, objects and complex numbers are not bit containers."""
    valid = np.array([0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="bits must be a boolean, integer or real"):
        bipolar_decode(_INADMISSIBLE_TYPES[case])
    with pytest.raises(ValueError, match="a must be a boolean, integer or real"):
        bipolar_multiply(_INADMISSIBLE_TYPES[case], valid)
    with pytest.raises(ValueError, match="b must be a boolean, integer or real"):
        bipolar_multiply(valid, _INADMISSIBLE_TYPES[case])


def test_nested_sequences_of_unequal_length_are_refused_by_name() -> None:
    """A ragged nested sequence is refused with a message that names the argument."""
    valid = np.array([0, 1, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="bits must be a rectangular array"):
        bipolar_decode([[0, 1], [1]])
    with pytest.raises(ValueError, match="a must be a rectangular array"):
        bipolar_multiply([[0, 1], [1]], valid)
    with pytest.raises(ValueError, match="b must be a rectangular array"):
        bipolar_multiply(valid, [[0, 1], [1]])


@pytest.mark.parametrize("empty", [[], np.zeros((0,), dtype=np.uint8), np.zeros((3, 0))])
def test_empty_input_is_refused(empty: ArrayLike) -> None:
    """An array without elements holds no bits, whatever its shape."""
    with pytest.raises(ValueError, match="bits must be non-empty"):
        bipolar_decode(empty)


@pytest.mark.parametrize("case", sorted(_BINARY_REPRESENTATIONS))
def test_every_exact_binary_representation_gives_the_same_result(case: str) -> None:
    """The result depends on the bits, not on the container that holds them."""
    bits_a, bits_b = _BINARY_REPRESENTATIONS[case]
    assert bipolar_decode(bits_a) == bipolar_decode(_REFERENCE_A)
    product = bipolar_multiply(bits_a, bits_b)
    assert product.dtype == np.uint8
    assert np.array_equal(product, bipolar_multiply(_REFERENCE_A, _REFERENCE_B))


def test_multiply_is_the_xnor_truth_table_in_any_shape() -> None:
    """Two-dimensional operands keep their shape and follow XNOR bit by bit."""
    product = bipolar_multiply(_REFERENCE_A, _REFERENCE_B)
    assert product.shape == _REFERENCE_A.shape
    assert product.tolist() == [[1, 0, 0, 1], [1, 0, 0, 1]]


def test_decode_of_known_bits() -> None:
    """Three ones in eight bits decode to 2 * 3/8 - 1."""
    assert bipolar_decode(np.array([[1, 0, 0, 0], [1, 0, 0, 1]])) == pytest.approx(-0.25)
    assert bipolar_decode(np.uint8(1)) == 1.0


def test_refused_input_is_left_unchanged_and_accepted_input_is_not_aliased() -> None:
    """Validation never writes to its input, and the result is a separate array."""
    fractional = np.array([0.9, 1.9])
    before = fractional.copy()
    with pytest.raises(ValueError):
        bipolar_decode(fractional)
    assert np.array_equal(fractional, before)
    operand = np.array([1, 0, 1], dtype=np.uint8)
    product = bipolar_multiply(operand, operand)
    product[0] = 0
    assert operand.tolist() == [1, 0, 1]
