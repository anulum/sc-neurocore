# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

"""Focused tests for failure analysis."""

from typing import Any

import pytest

from sc_neurocore.safety_cert.safety_cert import (
    FMEDA,
    FailureCategory,
    FailureMode,
    ReliabilityMetrics,
    SILLevel,
)


def _unsafe(value: object) -> Any:
    """Return a deliberately invalid runtime value for boundary tests."""
    return value


class TestFMEDA:
    def test_add_failure_mode(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "neuron", "stuck", FailureCategory.SAFE, 10.0)
        fmeda.add_failure_mode(fm)
        assert len(fmeda.failure_modes) == 1

    def test_add_failure_mode_rejects_duplicate_fm_id(self) -> None:
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("FM1", "neuron", "stuck", FailureCategory.SAFE, 10.0))
        with pytest.raises(ValueError, match="already exists"):
            fmeda.add_failure_mode(
                FailureMode("FM1", "neuron", "other", FailureCategory.SAFE, 12.0)
            )

    def test_add_failure_mode_rejects_invalid_contract(self) -> None:
        fmeda = FMEDA()
        with pytest.raises(ValueError, match="fm"):
            fmeda.add_failure_mode(_unsafe("bad"))

    def test_add_sc_standard_modes(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("sc_lif_neuron", acknowledge_synthetic_profile=True)
        assert len(fmeda.failure_modes) == 5

    def test_add_sc_standard_modes_normalises_component_whitespace(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes(" sc_lif_neuron ", acknowledge_synthetic_profile=True)
        assert all(fm.component == "sc_lif_neuron" for fm in fmeda.failure_modes)

    def test_add_sc_standard_modes_rejects_duplicate_component_seed(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        with pytest.raises(ValueError, match="already exists"):
            fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)

    def test_add_sc_standard_modes_rejects_invalid_component(self) -> None:
        fmeda = FMEDA()
        with pytest.raises(ValueError, match="component"):
            fmeda.add_sc_standard_modes("", acknowledge_synthetic_profile=True)

    def test_total_failure_rate(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        assert fmeda.total_failure_rate > 0

    def test_total_failure_rate_rejects_corrupted_internal_state(self) -> None:
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.total_failure_rate

    def test_total_failure_rate_rejects_corrupted_total_state(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.SAFE, 1.0)
        fm.failure_rate_fit = _unsafe(float("nan"))
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="failure_rate_fit"):
            _ = fmeda.total_failure_rate

    def test_safe_failure_fraction(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        sff = fmeda.safe_failure_fraction
        assert 0.0 < sff <= 1.0

    def test_safe_failure_fraction_rejects_corrupted_internal_state(self) -> None:
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.safe_failure_fraction

    def test_diagnostic_coverage(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        dc = fmeda.diagnostic_coverage
        assert 0.0 < dc <= 1.0

    def test_diagnostic_coverage_rejects_corrupted_internal_state(self) -> None:
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.diagnostic_coverage

    def test_diagnostic_coverage_rejects_corrupted_aggregate_state(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.failure_rate_fit = _unsafe(float("nan"))
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="denominator|weighted sum"):
            _ = fmeda.diagnostic_coverage

    def test_diagnostic_coverage_rejects_corrupted_entry_coverage_value(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = _unsafe(1.5)
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="entries must be in"):
            _ = fmeda.diagnostic_coverage

    def test_sff_by_component_rejects_corrupted_internal_state(self) -> None:
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            fmeda.sff_by_component()

    def test_sff_by_component_rejects_corrupted_component_totals(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = _unsafe(float("nan"))
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="totals"):
            fmeda.sff_by_component()

    def test_sff_by_component_rejects_empty_component_name(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.SAFE, 1.0)
        fm.component = _unsafe("")
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="component names"):
            fmeda.sff_by_component()

    def test_max_sil(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        sil = fmeda.max_achievable_sil()
        assert sil.value >= 1

    def test_max_sil_rejects_corrupted_coverage_state(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = _unsafe(float("nan"))
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="coverage|safe_failure_fraction"):
            fmeda.max_achievable_sil()

    def test_generate_report(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        report = fmeda.generate_report()
        assert "FMEDA" in report
        assert "FIT" in report

    def test_generate_report_orders_rows_by_failure_mode_id(self) -> None:
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("FM2", "n", "d", FailureCategory.SAFE, 1.0))
        fmeda.add_failure_mode(FailureMode("FM1", "n", "d", FailureCategory.SAFE, 1.0))
        lines = fmeda.generate_report().splitlines()
        rows = [
            line for line in lines if line.startswith("| FM") and (not line.startswith("| FM ID"))
        ]
        assert rows == sorted(rows)

    def test_generate_report_rejects_corrupted_internal_state(self) -> None:
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            fmeda.generate_report()

    def test_safe_failure_fraction_value(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "x", "safe", FailureCategory.SAFE, 100.0)
        fmeda.add_failure_mode(fm)
        assert fmeda.safe_failure_fraction == 1.0

    def test_sff_all_dangerous(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "x", "bad", FailureCategory.DANGEROUS_UNDETECTED, 100.0)
        fmeda.add_failure_mode(fm)
        assert fmeda.safe_failure_fraction == 0.0

    def test_residual_risk_rejects_corrupted_internal_state(self) -> None:
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.residual_risk_fit

    def test_residual_risk_rejects_corrupted_aggregate_state(self) -> None:
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = _unsafe(float("nan"))
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="residual_risk_fit"):
            _ = fmeda.residual_risk_fit

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"fm_id": ""}, "fm_id"),
            ({"component": ""}, "component"),
            ({"description": ""}, "description"),
            ({"category": "safe"}, "category"),
            ({"failure_rate_fit": -1.0}, "failure_rate_fit"),
            ({"failure_rate_fit": float("nan")}, "failure_rate_fit"),
            ({"failure_rate_fit": True}, "failure_rate_fit"),
            ({"diagnostic_coverage": -0.1}, "diagnostic_coverage"),
            ({"diagnostic_coverage": 1.1}, "diagnostic_coverage"),
            ({"diagnostic_coverage": float("inf")}, "diagnostic_coverage"),
            ({"diagnostic_coverage": False}, "diagnostic_coverage"),
            ({"mitigation": None}, "mitigation"),
        ],
    )
    def test_failure_mode_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "fm_id": "FM1",
            "component": "neuron",
            "description": "desc",
            "category": FailureCategory.SAFE,
            "failure_rate_fit": 1.0,
            "diagnostic_coverage": 0.5,
            "mitigation": "mitigate",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FailureMode(**_unsafe(values))


class TestResidualRisk:
    def test_residual_risk_all_safe(self) -> None:
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("FM1", "x", "safe", FailureCategory.SAFE, 100.0))
        assert fmeda.residual_risk_fit == 0.0

    def test_residual_risk_undetected(self) -> None:
        fmeda = FMEDA()
        fmeda.add_failure_mode(
            FailureMode("FM1", "x", "bad", FailureCategory.DANGEROUS_UNDETECTED, 100.0)
        )
        assert fmeda.residual_risk_fit == 100.0

    def test_residual_risk_partial(self) -> None:
        fmeda = FMEDA()
        fmeda.add_failure_mode(
            FailureMode("FM1", "x", "det", FailureCategory.DANGEROUS_DETECTED, 100.0, 0.9)
        )
        assert 0 < fmeda.residual_risk_fit < 100.0


class TestComponentSFF:
    def test_sff_by_component(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        fmeda.add_sc_standard_modes("encoder", acknowledge_synthetic_profile=True)
        sff_map = fmeda.sff_by_component()
        assert "neuron" in sff_map
        assert "encoder" in sff_map
        assert 0 < sff_map["neuron"] <= 1.0

    def test_sff_single_component(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("lif", acknowledge_synthetic_profile=True)
        sff_map = fmeda.sff_by_component()
        assert len(sff_map) == 1


class TestReliabilityMetrics:
    def test_mtbf(self) -> None:
        rm = ReliabilityMetrics(total_fit=100.0, dangerous_undetected_fit=5.0)
        assert rm.mtbf_hours > 0
        assert rm.mtbf_years > 0

    def test_pfh_d(self) -> None:
        rm = ReliabilityMetrics(total_fit=100.0, dangerous_undetected_fit=5.0)
        assert rm.pfh_d > 0

    def test_pfh_sil(self) -> None:
        rm = ReliabilityMetrics(total_fit=100.0, dangerous_undetected_fit=5.0)
        assert rm.pfh_sil.value >= 1

    def test_pfh_sil_rejects_corrupted_pfh_state(self) -> None:
        rm = ReliabilityMetrics(total_fit=100.0, dangerous_undetected_fit=5.0)
        rm.dangerous_undetected_fit = _unsafe(float("nan"))
        with pytest.raises(ValueError, match="pfh_d"):
            _ = rm.pfh_sil

    def test_from_fmeda(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        rm = ReliabilityMetrics.from_fmeda(fmeda)
        assert rm.total_fit > 0
        assert rm.mtbf_years > 0

    def test_from_fmeda_rejects_invalid_input(self) -> None:
        with pytest.raises(ValueError, match="fmeda"):
            ReliabilityMetrics.from_fmeda(_unsafe("bad"))

    def test_zero_fit(self) -> None:
        rm = ReliabilityMetrics(total_fit=0.0, dangerous_undetected_fit=0.0)
        assert rm.mtbf_hours == float("inf")
        assert rm.pfh_d == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"total_fit": -1.0}, "total_fit"),
            ({"total_fit": float("nan")}, "total_fit"),
            ({"total_fit": True}, "total_fit"),
            ({"dangerous_undetected_fit": -0.1}, "dangerous_undetected_fit"),
            ({"total_fit": 1.0, "dangerous_undetected_fit": 2.0}, "cannot exceed"),
            ({"dangerous_undetected_fit": float("inf")}, "dangerous_undetected_fit"),
            ({"dangerous_undetected_fit": False}, "dangerous_undetected_fit"),
        ],
    )
    def test_reliability_metrics_reject_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {"total_fit": 100.0, "dangerous_undetected_fit": 5.0}
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ReliabilityMetrics(**_unsafe(values))


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


class TestSyntheticProfileBoundary:
    def test_profile_requires_explicit_boolean_acknowledgement(self) -> None:
        fmeda = FMEDA()
        with pytest.raises(ValueError, match="must be a boolean"):
            fmeda.add_sc_standard_modes(
                "neuron",
                acknowledge_synthetic_profile=_unsafe("yes"),
            )
        with pytest.raises(ValueError, match="requires acknowledge"):
            fmeda.add_sc_standard_modes("neuron")

    def test_empty_fmeda_report_is_explicitly_unassessed(self) -> None:
        assert "Status: not assessed" in FMEDA().generate_report()
