# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Fault-injection model validation tests

"""Validation tests for fault-injection data models and reports."""

from __future__ import annotations

from fault_injection_support import *  # noqa: F403


def test_radiation_profile_rejects_non_numeric_ber():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    with pytest.raises(ValueError, match="ber must be"):
        RadiationProfile(name="bad", ber="high")  # type: ignore[arg-type]


def test_fault_injection_result_rejects_non_integer_field():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    with pytest.raises(ValueError, match="must be an integer"):
        FaultInjectionResult(
            original_popcount="x",  # type: ignore[arg-type]
            corrupted_popcount=0,
            bits_flipped=0,
            bitstream_length=10,
        )


def test_resilience_report_rejects_empty_fault_model():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    with pytest.raises(ValueError, match="fault_model must be a non-empty string"):
        ResilienceReport(**{**_VALID_REPORT, "fault_model": "   "})


def test_resilience_report_rejects_non_numeric_field():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    with pytest.raises(ValueError, match="ber must be numeric"):
        ResilienceReport(**{**_VALID_REPORT, "ber": "x"})


def test_resilience_report_rejects_non_finite_field():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    with pytest.raises(ValueError, match="mean_error must be finite"):
        ResilienceReport(**{**_VALID_REPORT, "mean_error": float("inf")})
