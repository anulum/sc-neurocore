# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning scalar float validation tests

"""Real-number, finiteness, bounded-domain, seed, and saturation contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore._native import learning_validation as validation


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
