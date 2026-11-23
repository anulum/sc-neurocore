# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Fault-injection model validation tests

"""Validation tests for fault-injection data models and reports."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fault_injection_support import (
    _VALID_REPORT,
)
from sc_neurocore.fault_injection.fault_injection import (
    FaultInjectionResult,
    RadiationProfile,
    ResilienceReport,
)


def test_radiation_profile_rejects_non_numeric_ber() -> None:
    """A radiation profile refuses a text bit-error rate."""
    with pytest.raises(ValueError, match="ber must be"):
        RadiationProfile(name="bad", ber=cast(Any, "high"))


def test_fault_injection_result_rejects_non_integer_field() -> None:
    """A fault-injection result refuses a popcount that is not an integer."""
    with pytest.raises(ValueError, match="must be an integer"):
        FaultInjectionResult(
            original_popcount=cast(Any, "x"),
            corrupted_popcount=0,
            bits_flipped=0,
            bitstream_length=10,
        )


def test_resilience_report_rejects_empty_fault_model() -> None:
    """A resilience report refuses a blank fault-model name."""
    with pytest.raises(ValueError, match="fault_model must be a non-empty string"):
        ResilienceReport(**cast(dict[str, Any], {**_VALID_REPORT, "fault_model": "   "}))


def test_resilience_report_rejects_non_numeric_field() -> None:
    """A resilience report refuses a text bit-error rate."""
    with pytest.raises(ValueError, match="ber must be numeric"):
        ResilienceReport(**cast(dict[str, Any], {**_VALID_REPORT, "ber": "x"}))


def test_resilience_report_rejects_non_finite_field() -> None:
    """A resilience report refuses an infinite mean error."""
    with pytest.raises(ValueError, match="mean_error must be finite"):
        ResilienceReport(**cast(dict[str, Any], {**_VALID_REPORT, "mean_error": float("inf")}))
