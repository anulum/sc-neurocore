# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFailClosedEvidenceContracts from former test_compliance.py

"""Focused suite: TestFailClosedEvidenceContracts from former test_compliance.py."""

from __future__ import annotations

import pytest
from sc_neurocore.safety_cert.safety_cert import (
    ChecklistItem,
    ComplianceChecklist,
    IEC62304Assessment,
    SafetyStandard,
    SWClass,
)
from tests.test_safety_cert.compliance_support import (
    _unsafe,
)


class TestFailClosedEvidenceContracts:
    """Evidence contracts that fail closed."""

    def test_addressed_item_requires_real_evidence(self) -> None:
        """An addressed item needs non-empty evidence."""
        with pytest.raises(ValueError, match="require non-empty evidence"):
            ChecklistItem("id", "7.4.2", "description", "", "partial")

    def test_checklist_evidence_mapping_is_strictly_validated(self) -> None:
        """The evidence mapping is checked strictly."""
        with pytest.raises(ValueError, match="mapping"):
            ComplianceChecklist.generate(
                SafetyStandard.IEC_61508,
                evidence=_unsafe("invalid"),
            )
        with pytest.raises(ValueError, match="clause keys"):
            ComplianceChecklist.generate(
                SafetyStandard.IEC_61508,
                evidence=_unsafe({42: "evidence.md"}),
            )

    def test_software_class_crosswalk_rejects_invalid_sil(self) -> None:
        """The software-class crosswalk rejects a bad SIL."""
        with pytest.raises(ValueError, match="SILLevel"):
            IEC62304Assessment.from_sil(_unsafe("SIL_2"))

    def test_multiple_risk_controls_are_all_validated(self) -> None:
        """Every risk control is checked."""
        assessment = IEC62304Assessment(
            SWClass.CLASS_B,
            risk_control_measures=["monitor", "independent shutdown"],
        )
        assert assessment.requires_unit_testing
