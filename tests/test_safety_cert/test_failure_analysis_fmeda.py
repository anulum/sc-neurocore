# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFMEDA from former test_failure_analysis.py

"""Focused suite: TestFMEDA from former test_failure_analysis.py."""

from __future__ import annotations

import pytest
from sc_neurocore.safety_cert.safety_cert import (
    FailureCategory,
    FailureMode,
    FMEDA,
)
from tests.test_safety_cert.failure_analysis_support import (
    _unsafe,
)
from typing import (
    Any,
)


class TestFMEDA:
    """Failure modes, effects and diagnostic analysis."""

    def test_add_failure_mode(self) -> None:
        """An added failure mode is kept."""
        fmeda = FMEDA()
        fm = FailureMode("FM1", "neuron", "stuck", FailureCategory.SAFE, 10.0)
        fmeda.add_failure_mode(fm)
        assert len(fmeda.failure_modes) == 1

    def test_add_failure_mode_rejects_duplicate_fm_id(self) -> None:
        """A duplicate failure mode id is rejected."""
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("FM1", "neuron", "stuck", FailureCategory.SAFE, 10.0))
        with pytest.raises(ValueError, match="already exists"):
            fmeda.add_failure_mode(
                FailureMode("FM1", "neuron", "other", FailureCategory.SAFE, 12.0)
            )

    def test_add_failure_mode_rejects_invalid_contract(self) -> None:
        """A malformed failure mode is rejected."""
        fmeda = FMEDA()
        with pytest.raises(ValueError, match="fm"):
            fmeda.add_failure_mode(_unsafe("bad"))

    def test_add_sc_standard_modes(self) -> None:
        """The standard SC modes add five failure modes."""
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("sc_lif_neuron", acknowledge_synthetic_profile=True)
        assert len(fmeda.failure_modes) == 5

    def test_add_sc_standard_modes_normalises_component_whitespace(self) -> None:
        """The component name is trimmed."""
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes(" sc_lif_neuron ", acknowledge_synthetic_profile=True)
        assert all(fm.component == "sc_lif_neuron" for fm in fmeda.failure_modes)

    def test_add_sc_standard_modes_rejects_duplicate_component_seed(self) -> None:
        """Adding the same component twice is rejected."""
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        with pytest.raises(ValueError, match="already exists"):
            fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)

    def test_add_sc_standard_modes_rejects_invalid_component(self) -> None:
        """An invalid component name is rejected."""
        fmeda = FMEDA()
        with pytest.raises(ValueError, match="component"):
            fmeda.add_sc_standard_modes("", acknowledge_synthetic_profile=True)

    def test_total_failure_rate(self) -> None:
        """The total failure rate is positive."""
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        assert fmeda.total_failure_rate > 0

    def test_total_failure_rate_rejects_corrupted_internal_state(self) -> None:
        """Corrupted internal state is rejected."""
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.total_failure_rate

    def test_total_failure_rate_rejects_corrupted_total_state(self) -> None:
        """A corrupted failure rate is rejected."""
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.SAFE, 1.0)
        fm.failure_rate_fit = _unsafe(float("nan"))
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="failure_rate_fit"):
            _ = fmeda.total_failure_rate

    def test_safe_failure_fraction(self) -> None:
        """The SFF lies in (0, 1]."""
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        sff = fmeda.safe_failure_fraction
        assert 0.0 < sff <= 1.0

    def test_safe_failure_fraction_rejects_corrupted_internal_state(self) -> None:
        """Corrupted internal state is rejected."""
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.safe_failure_fraction

    def test_diagnostic_coverage(self) -> None:
        """The diagnostic coverage lies in (0, 1]."""
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        dc = fmeda.diagnostic_coverage
        assert 0.0 < dc <= 1.0

    def test_diagnostic_coverage_rejects_corrupted_internal_state(self) -> None:
        """Corrupted internal state is rejected."""
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.diagnostic_coverage

    def test_diagnostic_coverage_rejects_corrupted_aggregate_state(self) -> None:
        """A corrupted aggregate is rejected."""
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.failure_rate_fit = _unsafe(float("nan"))
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="denominator|weighted sum"):
            _ = fmeda.diagnostic_coverage

    def test_diagnostic_coverage_rejects_corrupted_entry_coverage_value(self) -> None:
        """A corrupted entry coverage is rejected."""
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = _unsafe(1.5)
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="entries must be in"):
            _ = fmeda.diagnostic_coverage

    def test_sff_by_component_rejects_corrupted_internal_state(self) -> None:
        """Corrupted internal state is rejected."""
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            fmeda.sff_by_component()

    def test_sff_by_component_rejects_corrupted_component_totals(self) -> None:
        """Corrupted component totals are rejected."""
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = _unsafe(float("nan"))
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="totals"):
            fmeda.sff_by_component()

    def test_sff_by_component_rejects_empty_component_name(self) -> None:
        """An empty component name is rejected."""
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.SAFE, 1.0)
        fm.component = _unsafe("")
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="component names"):
            fmeda.sff_by_component()

    def test_max_sil(self) -> None:
        """The analysis supports at least SIL 1."""
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        sil = fmeda.max_achievable_sil()
        assert sil.value >= 1

    def test_max_sil_rejects_corrupted_coverage_state(self) -> None:
        """A corrupted coverage is rejected."""
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = _unsafe(float("nan"))
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="coverage|safe_failure_fraction"):
            fmeda.max_achievable_sil()

    def test_generate_report(self) -> None:
        """The report names FMEDA and FIT."""
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        report = fmeda.generate_report()
        assert "FMEDA" in report
        assert "FIT" in report

    def test_generate_report_orders_rows_by_failure_mode_id(self) -> None:
        """Report rows are ordered by failure mode id."""
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("FM2", "n", "d", FailureCategory.SAFE, 1.0))
        fmeda.add_failure_mode(FailureMode("FM1", "n", "d", FailureCategory.SAFE, 1.0))
        lines = fmeda.generate_report().splitlines()
        rows = [
            line for line in lines if line.startswith("| FM") and (not line.startswith("| FM ID"))
        ]
        assert rows == sorted(rows)

    def test_generate_report_rejects_corrupted_internal_state(self) -> None:
        """Corrupted internal state is rejected."""
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            fmeda.generate_report()

    def test_safe_failure_fraction_value(self) -> None:
        """Only safe failures give an SFF of 1."""
        fmeda = FMEDA()
        fm = FailureMode("FM1", "x", "safe", FailureCategory.SAFE, 100.0)
        fmeda.add_failure_mode(fm)
        assert fmeda.safe_failure_fraction == 1.0

    def test_sff_all_dangerous(self) -> None:
        """Only dangerous failures give an SFF of 0."""
        fmeda = FMEDA()
        fm = FailureMode("FM1", "x", "bad", FailureCategory.DANGEROUS_UNDETECTED, 100.0)
        fmeda.add_failure_mode(fm)
        assert fmeda.safe_failure_fraction == 0.0

    def test_residual_risk_rejects_corrupted_internal_state(self) -> None:
        """Corrupted internal state is rejected."""
        fmeda = FMEDA()
        fmeda.failure_modes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.residual_risk_fit

    def test_residual_risk_rejects_corrupted_aggregate_state(self) -> None:
        """A corrupted residual risk is rejected."""
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
        """A failure mode rejects each invalid field."""
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
