# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundaryContracts from former test_certification.py

"""Focused suite: TestBoundaryContracts from former test_certification.py."""

from __future__ import annotations

from tests.test_safety_cert.certification_support import *  # noqa: F403

class TestBoundaryContracts:
    def test_empty_checklist_package_reports_zero_fraction(self) -> None:
        pkg = CertificationPackage(
            standard=SafetyStandard.IEC_61508,
            sil_level=SILLevel.SIL_2,
            traceability_report="trace",
            fmeda_report="fmeda",
            formal_cert_report="formal",
            wcet_report="wcet",
            checklist=[],
        )
        assert pkg.checklist_coverage == 0.0

    def test_package_revalidates_clause_after_status_contract(self) -> None:
        item = ChecklistItem("IEC 61508_7.4.2", "7.4.2", "desc", "formal/", "partial")
        item.clause = ""
        with pytest.raises(ValueError, match="checklist clauses"):
            CertificationPackage(
                standard=SafetyStandard.IEC_61508,
                sil_level=SILLevel.SIL_2,
                traceability_report="t",
                fmeda_report="f",
                formal_cert_report="p",
                wcet_report="w",
                checklist=[item],
            )

    def test_certification_generator_rejects_formal_property_id_whitespace(self) -> None:
        prop = FormalProperty(" P1", "neuron", "desc", "assert", "proven")
        with pytest.raises(ValueError, match="prop_id"):
            CertificationGenerator().generate(
                SafetyStandard.IEC_61508, SILLevel.SIL_2, ["neuron"], [prop]
            )

    def test_certification_generator_rejects_boolean_clock_configuration(self) -> None:
        prop = FormalProperty("P1", "neuron", "desc", "assert", "proven")
        with pytest.raises(ValueError, match="clock_mhz"):
            CertificationGenerator().generate(
                SafetyStandard.IEC_61508,
                SILLevel.SIL_2,
                ["neuron"],
                [prop],
                network_config={"clock_mhz": True},
            )
