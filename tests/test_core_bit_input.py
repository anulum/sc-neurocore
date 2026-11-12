# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One validated bit conversion for the core package

"""Bind the single validated conversion of caller input to one byte per bit.

The cross-correlation diagnostics and the bipolar primitives use the same
function; each states the shape rule it needs.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.core import bipolar, estimate_scc, sc_correlation
from sc_neurocore.core.bit_input import as_bit_array

_REFERENCE_A = np.array([[1, 0, 1, 1], [0, 0, 1, 0]], dtype=np.uint8)
_REFERENCE_B = np.array([[1, 1, 0, 1], [0, 1, 0, 0]], dtype=np.uint8)


def test_one_conversion_serves_both_core_modules() -> None:
    """Neither module keeps a private copy of the conversion."""
    assert not hasattr(bipolar, "_as_bit_array")
    assert not hasattr(sc_correlation, "_as_bit_array")
    assert vars(bipolar)["as_bit_array"] is as_bit_array
    assert vars(sc_correlation)["as_bit_array"] is as_bit_array


def test_shape_rule_is_the_callers() -> None:
    """The diagnostics need a stream; the bipolar primitives take any shape."""
    assert as_bit_array(_REFERENCE_A, "bits").shape == (2, 4)
    with pytest.raises(ValueError, match="bits must be one-dimensional"):
        as_bit_array(_REFERENCE_A, "bits", one_dimensional=True)
    with pytest.raises(ValueError, match="bits_a must be one-dimensional"):
        estimate_scc(_REFERENCE_A, _REFERENCE_B)
    with pytest.raises(ValueError, match="bits must be one-dimensional"):
        as_bit_array([[0, 1], [1]], "bits", one_dimensional=True)
    assert as_bit_array([True, False], "bits", one_dimensional=True).tolist() == [1, 0]


@pytest.mark.parametrize("one_dimensional", [False, True])
def test_result_never_shares_memory_with_the_input(one_dimensional: bool) -> None:
    """Input that already holds one byte per bit is copied, not returned."""
    bits = np.array([1, 0, 1], dtype=np.uint8)
    result = as_bit_array(bits, "bits", one_dimensional=one_dimensional)
    assert result is not bits
    assert not np.shares_memory(result, bits)
    result[0] = 0
    assert bits.tolist() == [1, 0, 1]
