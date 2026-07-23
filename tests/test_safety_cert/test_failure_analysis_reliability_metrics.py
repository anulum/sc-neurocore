# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReliabilityMetrics from former test_failure_analysis.py

"""Focused suite: TestReliabilityMetrics from former test_failure_analysis.py."""

from __future__ import annotations

from tests.test_safety_cert.failure_analysis_support import *  # noqa: F403

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
