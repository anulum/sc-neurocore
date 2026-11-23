# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestComplianceChecklist from former test_compliance.py

"""Focused suite: TestComplianceChecklist from former test_compliance.py."""

from __future__ import annotations

import pytest
from sc_neurocore.safety_cert.safety_cert import (
    ChecklistItem,
    ComplianceChecklist,
    SafetyStandard,
)
from tests.test_safety_cert.compliance_support import (
    _unsafe,
)
from typing import (
    Any,
)


class TestComplianceChecklist:
    """Compliance checklists for each standard."""

    def test_iec_61508(self) -> None:
        """IEC 61508 has seven checklist items."""
        items = ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        assert len(items) == 7
        assert all(isinstance(i, ChecklistItem) for i in items)

    def test_iso_26262(self) -> None:
        """ISO 26262 has seven checklist items."""
        items = ComplianceChecklist.generate(SafetyStandard.ISO_26262)
        assert len(items) == 7

    def test_fda_class_iii(self) -> None:
        """FDA class III has seven checklist items."""
        items = ComplianceChecklist.generate(SafetyStandard.FDA_CLASS_III)
        assert len(items) == 7

    def test_do_254(self) -> None:
        """DO-254 has six checklist items."""
        items = ComplianceChecklist.generate(SafetyStandard.DO_254)
        assert len(items) == 6

    def test_en_50129(self) -> None:
        """EN 50129 has six checklist items."""
        items = ComplianceChecklist.generate(SafetyStandard.EN_50129)
        assert len(items) == 6

    def test_items_have_evidence(self) -> None:
        """New items start without evidence and unaddressed."""
        items = ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        assert all(not item.evidence and item.status == "not_addressed" for item in items)

    def test_generate_rejects_invalid_standard(self) -> None:
        """An unknown standard is rejected."""
        with pytest.raises(ValueError, match="standard"):
            ComplianceChecklist.generate(_unsafe("IEC 61508"))

    def test_generate_rejects_duplicate_clause_definitions(self) -> None:
        """Duplicate clause definitions are rejected."""
        original = ComplianceChecklist.IEC_61508_CLAUSES
        try:
            ComplianceChecklist.IEC_61508_CLAUSES = [
                ("7.4.2", "A", "formal/"),
                ("7.4.2", "B", "formal/"),
            ]
            with pytest.raises(ValueError, match="duplicates"):
                ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        finally:
            ComplianceChecklist.IEC_61508_CLAUSES = original

    def test_generate_rejects_corrupted_clause_definition_shape(self) -> None:
        """A malformed clause definition is rejected."""
        original = ComplianceChecklist.IEC_61508_CLAUSES
        try:
            ComplianceChecklist.IEC_61508_CLAUSES = [("7.4.2", "A", "formal/"), ("7.4.3", "B", "")]
            with pytest.raises(ValueError, match="clause definitions"):
                ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        finally:
            ComplianceChecklist.IEC_61508_CLAUSES = original

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"item_id": ""}, "item_id"),
            ({"clause": ""}, "clause"),
            ({"description": ""}, "description"),
            ({"evidence": None}, "evidence"),
            ({"status": "ok"}, "status"),
        ],
    )
    def test_checklist_item_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        """A checklist item rejects each invalid field."""
        values = {
            "item_id": "IEC 61508_7.4.2",
            "clause": "7.4.2",
            "description": "Formal verification of safety functions",
            "evidence": "formal/",
            "status": "partial",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ChecklistItem(**_unsafe(values))
