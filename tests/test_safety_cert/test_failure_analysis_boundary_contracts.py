# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundaryContracts from former test_failure_analysis.py

"""Focused suite: TestBoundaryContracts from former test_failure_analysis.py."""

from __future__ import annotations

from tests.test_safety_cert.failure_analysis_support import *  # noqa: F403

class TestBoundaryContracts:
    def test_fmeda_rejects_nonfinite_aggregate_total(self) -> None:
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("F1", "core", "desc", FailureCategory.SAFE, 1e308, 1.0))
        fmeda.add_failure_mode(FailureMode("F2", "core", "desc", FailureCategory.SAFE, 1e308, 1.0))
        with pytest.raises(ValueError, match="total_failure_rate"):
            _ = fmeda.total_failure_rate

    def test_fmeda_empty_and_zero_rate_paths_are_bounded(self) -> None:
        empty = FMEDA()
        assert empty.safe_failure_fraction == 0.0
        zero = FMEDA()
        zero.add_failure_mode(FailureMode("F1", "core", "desc", FailureCategory.SAFE, 0.0, 1.0))
        assert zero.safe_failure_fraction == 0.0
        detected_zero = FMEDA()
        detected_zero.add_failure_mode(
            FailureMode("F2", "core", "desc", FailureCategory.DANGEROUS_DETECTED, 0.0, 0.9)
        )
        with pytest.raises(ValueError, match="denominator"):
            _ = detected_zero.diagnostic_coverage

    def test_fmeda_component_sff_rejects_nonfinite_component_total(self) -> None:
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("F1", "core", "desc", FailureCategory.SAFE, 1e308, 1.0))
        fmeda.add_failure_mode(FailureMode("F2", "core", "desc", FailureCategory.SAFE, 1e308, 1.0))
        with pytest.raises(ValueError, match="component failure-rate totals"):
            fmeda.sff_by_component()

    def test_fmeda_sil_threshold_boundaries(self, monkeypatch: Any) -> None:
        sil4 = FMEDA()
        sil4.add_failure_mode(
            FailureMode("F1", "core", "desc", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.99)
        )
        assert sil4.max_achievable_sil() == SILLevel.SIL_4
        sil3 = FMEDA()
        sil3.add_failure_mode(
            FailureMode("F2", "core", "desc", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.99)
        )
        sil3.add_failure_mode(
            FailureMode("F3", "core", "desc", FailureCategory.DANGEROUS_UNDETECTED, 0.0102, 0.0)
        )
        assert sil3.max_achievable_sil() == SILLevel.SIL_3
        sil1 = FMEDA()
        sil1.add_failure_mode(
            FailureMode("F4", "core", "desc", FailureCategory.DANGEROUS_UNDETECTED, 1.0, 0.0)
        )
        assert sil1.max_achievable_sil() == SILLevel.SIL_1
        monkeypatch.setattr(FMEDA, "safe_failure_fraction", property(lambda self: float("nan")))
        with pytest.raises(ValueError, match="safe_failure_fraction"):
            sil4.max_achievable_sil()

    @pytest.mark.parametrize(
        ("dangerous_undetected_fit", "expected"),
        [
            (10.0, SILLevel.SIL_4),
            (100.0, SILLevel.SIL_3),
            (1000.0, SILLevel.SIL_2),
            (2000.0, SILLevel.SIL_1),
        ],
    )
    def test_reliability_pfh_sil_thresholds(
        self, dangerous_undetected_fit: Any, expected: Any
    ) -> None:
        metrics = ReliabilityMetrics(
            total_fit=dangerous_undetected_fit, dangerous_undetected_fit=dangerous_undetected_fit
        )
        assert metrics.pfh_sil == expected
