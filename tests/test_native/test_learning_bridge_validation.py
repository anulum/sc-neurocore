# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native-learning validation boundary tests

"""Direct fail-closed tests for autonomous-learning ABI validation."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore._native.learning_validation import (
    MAX_U32,
    as_bool_vector,
    as_float_vector,
    require_count,
    require_finite_float,
)


def test_count_rejects_values_above_native_u32_domain() -> None:
    """Layer sizes exceeding the native ABI must fail before FFI dispatch."""
    with pytest.raises(ValueError, match=rf"count must be <= {MAX_U32}"):
        require_count(MAX_U32 + 1)


@pytest.mark.parametrize("value", [True, object()])
def test_finite_float_rejects_boolean_and_non_real_values(value: object) -> None:
    """Boolean and non-real scalars must not cross a floating-point ABI."""
    with pytest.raises(TypeError, match="must be a real number"):
        require_finite_float(name="reward", value=value)


def test_bool_vector_accepts_explicit_binary_numbers() -> None:
    """Numeric Boolean input is accepted only when every value is binary."""
    result = as_bool_vector([0, 1], name="spikes")

    np.testing.assert_array_equal(result, np.array([False, True]))
    assert result.flags.c_contiguous


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["not-binary"], "booleans or binary numbers"),
        ([0.0, np.nan], "boolean, 0, or 1"),
        ([0.0, 2.0], "boolean, 0, or 1"),
    ],
)
def test_bool_vector_rejects_unconvertible_nonfinite_and_nonbinary_values(
    values: list[object],
    message: str,
) -> None:
    """Every conversion and binary-domain failure must remain fail-closed."""
    exception = TypeError if isinstance(values[0], str) else ValueError
    with pytest.raises(exception, match=message):
        as_bool_vector(values, name="spikes")


def test_float_vector_rejects_values_that_cannot_convert_to_float32() -> None:
    """Non-numeric vector content must fail with the public type contract."""
    with pytest.raises(TypeError, match="must contain real numeric values"):
        as_float_vector(["not-numeric"], name="reward")
