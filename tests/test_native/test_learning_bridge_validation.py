# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning validation tests

"""Exhaustive fail-closed tests for scalar and vector ABI validation."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore._native import learning_validation as validation


def test_integral_validators_accept_integer_domains() -> None:
    assert validation.require_integral(name="value", value=np.int64(3)) == 3
    assert validation.require_non_negative_integral(name="value", value=0) == 0
    assert validation.require_integral_range(name="value", value=4, lower=2, upper=4) == 4
    assert validation.require_count(1) == 1
    assert validation.require_rule_type(validation.RULE_BCM) == validation.RULE_BCM


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_integral_validator_rejects_non_integer(value: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        validation.require_integral(name="value", value=value)


def test_unsigned_and_range_domains_reject_bounds() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        validation.require_non_negative_integral(name="value", value=-1)
    with pytest.raises(ValueError, match="2..=4"):
        validation.require_integral_range(name="value", value=1, lower=2, upper=4)
    with pytest.raises(ValueError, match="2..=4"):
        validation.require_integral_range(name="value", value=5, lower=2, upper=4)


@pytest.mark.parametrize("value", [0, -1])
def test_count_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError, match="> 0"):
        validation.require_count(value)


def test_count_and_rule_reject_unsupported_values() -> None:
    with pytest.raises(ValueError, match="count must be <="):
        validation.require_count(validation.MAX_U32 + 1)
    with pytest.raises(ValueError, match="one of"):
        validation.require_rule_type(99)


def test_boolean_validator_is_exact() -> None:
    assert validation.require_bool(name="flag", value=True) is True
    with pytest.raises(TypeError, match="must be bool"):
        validation.require_bool(name="flag", value=1)


def test_float_validators_accept_their_domains() -> None:
    assert validation.require_finite_float(name="value", value=np.float32(1.25)) == 1.25
    assert validation.require_positive_float(name="value", value=0.1) == 0.1
    assert validation.require_non_negative_float(name="value", value=0) == 0.0
    assert validation.require_unit_interval(name="value", value=1) == 1.0


@pytest.mark.parametrize("value", [True, "1.0", object()])
def test_float_validator_rejects_non_real(value: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        validation.require_finite_float(name="value", value=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_float_validator_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        validation.require_finite_float(name="value", value=value)


def test_float_domain_validators_reject_bounds() -> None:
    with pytest.raises(ValueError, match="> 0"):
        validation.require_positive_float(name="value", value=0.0)
    with pytest.raises(ValueError, match=">= 0"):
        validation.require_non_negative_float(name="value", value=-0.1)
    for value in (-0.1, 1.1):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validation.require_unit_interval(name="value", value=value)


def test_seed_and_saturation_helpers() -> None:
    assert validation.require_u32_seed(name="seed", value=validation.MAX_U32) == validation.MAX_U32
    assert validation.require_u64_seed(name="seed", value=validation.MAX_U64) == validation.MAX_U64
    with pytest.raises(ValueError):
        validation.require_u32_seed(name="seed", value=validation.MAX_U32 + 1)
    with pytest.raises(ValueError):
        validation.require_u64_seed(name="seed", value=-1)
    assert validation.saturate(-2, -1, 1) == -1
    assert validation.saturate(0, -1, 1) == 0
    assert validation.saturate(2, -1, 1) == 1


def test_bool_vectors_preserve_binary_semantics() -> None:
    direct = validation.as_bool_vector(np.array([True, False]), name="flags", length=2)
    numeric = validation.as_bool_vector([0, 1], name="flags")
    assert direct.dtype == np.bool_ and direct.flags.c_contiguous
    assert numeric.tolist() == [False, True]


@pytest.mark.parametrize("values", [[0, 2], [0.0, np.nan]])
def test_bool_vectors_reject_non_binary_values(values: object) -> None:
    with pytest.raises(ValueError, match="boolean, 0, or 1"):
        validation.as_bool_vector(values, name="flags")


def test_bool_vectors_reject_non_numeric_values() -> None:
    with pytest.raises(TypeError, match="booleans or binary"):
        validation.as_bool_vector([object()], name="flags")


def test_vector_shape_and_length_are_exact() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validation.as_bool_vector([[True]], name="flags")
    with pytest.raises(ValueError, match="length 2"):
        validation.as_float_vector([1.0], name="values", length=2)


def test_float_vectors_are_finite_contiguous_float32() -> None:
    result = validation.as_float_vector(np.array([1, 2], dtype=np.int64), name="values")
    assert result.dtype == np.float32 and result.flags.c_contiguous
    with pytest.raises(TypeError, match="numeric"):
        validation.as_float_vector([object()], name="values")
    with pytest.raises(ValueError, match="finite"):
        validation.as_float_vector([np.inf], name="values")


def test_probability_vectors_enforce_closed_unit_interval() -> None:
    result = validation.as_probability_vector([0.0, 1.0], name="probabilities")
    assert result.tolist() == [0.0, 1.0]
    for values in ([-0.1], [1.1]):
        with pytest.raises(ValueError, match="probabilities"):
            validation.as_probability_vector(values, name="probabilities")
