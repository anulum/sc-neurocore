# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning integral validation tests

"""Integral, range, count, rule-type, and exact-boolean validation contracts."""

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
