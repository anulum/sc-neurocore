# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

"""Focused tests for compliance."""

from typing import Any

import pytest

from sc_neurocore.safety_cert.safety_cert import (
    CROSS_MAP,
    ChecklistItem,
    ComplianceChecklist,
    CrossStandardMapper,
    IEC62304Assessment,
    SafetyStandard,
    SILLevel,
    SWClass,
)


def _unsafe(value: object) -> Any:
    """Return a deliberately invalid runtime value for boundary tests."""
    return value


class TestComplianceChecklist:
    def test_iec_61508(self) -> None:
        items = ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        assert len(items) == 7
        assert all(isinstance(i, ChecklistItem) for i in items)

    def test_iso_26262(self) -> None:
        items = ComplianceChecklist.generate(SafetyStandard.ISO_26262)
        assert len(items) == 7

    def test_fda_class_iii(self) -> None:
        items = ComplianceChecklist.generate(SafetyStandard.FDA_CLASS_III)
        assert len(items) == 7

    def test_do_254(self) -> None:
        items = ComplianceChecklist.generate(SafetyStandard.DO_254)
        assert len(items) == 6

    def test_en_50129(self) -> None:
        items = ComplianceChecklist.generate(SafetyStandard.EN_50129)
        assert len(items) == 6

    def test_items_have_evidence(self) -> None:
        items = ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        assert all(not item.evidence and item.status == "not_addressed" for item in items)

    def test_generate_rejects_invalid_standard(self) -> None:
        with pytest.raises(ValueError, match="standard"):
            ComplianceChecklist.generate(_unsafe("IEC 61508"))

    def test_generate_rejects_duplicate_clause_definitions(self) -> None:
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


class TestIEC62304:
    def test_from_sil_1(self) -> None:
        a = IEC62304Assessment.from_sil(SILLevel.SIL_1)
        assert a.sw_class == SWClass.CLASS_A
        assert not a.requires_unit_testing

    def test_from_sil_3(self) -> None:
        a = IEC62304Assessment.from_sil(SILLevel.SIL_3)
        assert a.sw_class == SWClass.CLASS_C
        assert a.requires_unit_testing
        assert a.requires_architectural_design

    def test_class_b(self) -> None:
        a = IEC62304Assessment(sw_class=SWClass.CLASS_B)
        assert a.requires_unit_testing
        assert not a.requires_architectural_design

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sw_class": "B"}, "sw_class"),
            ({"hazard_description": None}, "hazard_description"),
            ({"risk_control_measures": "measure"}, "risk_control_measures"),
            ({"risk_control_measures": ["", "measure"]}, "risk_control_measures"),
        ],
    )
    def test_iec62304_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "sw_class": SWClass.CLASS_B,
            "hazard_description": "hazard",
            "risk_control_measures": ["measure 1"],
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            IEC62304Assessment(**_unsafe(values))


class TestCrossStandardMapper:
    def test_equivalent_clauses(self) -> None:
        equiv = CrossStandardMapper.equivalent_clauses("IEC 61508", "7.4.2")
        assert len(equiv) == 2
        assert ("ISO 26262", "6.7.4") in equiv

    def test_equivalent_clauses_normalises_whitespace(self) -> None:
        equiv = CrossStandardMapper.equivalent_clauses(" IEC 61508 ", " 7.4.2 ")
        assert ("ISO 26262", "6.7.4") in equiv

    def test_no_mapping(self) -> None:
        equiv = CrossStandardMapper.equivalent_clauses("IEC 61508", "99.99")
        assert equiv == []

    def test_coverage_overlap_rejects_malformed_item_id(self) -> None:
        bad_item = ChecklistItem(
            item_id="MALFORMED",
            clause="7.4.2",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        good_item = ChecklistItem(
            item_id="ISO 26262_6.7.4",
            clause="6.7.4",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        with pytest.raises(ValueError, match="item_id"):
            CrossStandardMapper.coverage_overlap([bad_item], [good_item])

    def test_coverage_overlap_rejects_malformed_right_item_id(self) -> None:
        good_item = ChecklistItem(
            item_id="IEC 61508_7.4.2",
            clause="7.4.2",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        bad_item = ChecklistItem(
            item_id="MALFORMED",
            clause="6.7.4",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        with pytest.raises(ValueError, match="checklist_b"):
            CrossStandardMapper.coverage_overlap([good_item], [bad_item])

    def test_coverage_overlap_rejects_corrupted_clause_state(self) -> None:
        left = ChecklistItem(
            item_id="IEC 61508_7.4.2",
            clause="7.4.2",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        right = ChecklistItem(
            item_id="ISO 26262_6.7.4",
            clause="6.7.4",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        right.clause = _unsafe("")
        with pytest.raises(ValueError, match="clauses"):
            CrossStandardMapper.coverage_overlap([left], [right])

    def test_coverage_overlap_rejects_corrupted_status_state(self) -> None:
        left = ChecklistItem("IEC 61508_7.4.2", "7.4.2", "desc", "formal/", "partial")
        right = ChecklistItem("ISO 26262_6.7.4", "6.7.4", "desc", "formal/", "partial")
        left.status = _unsafe("bad")
        with pytest.raises(ValueError, match="statuses"):
            CrossStandardMapper.coverage_overlap([left], [right])

    def test_coverage_overlap_deduplicates_equivalent_mappings(self) -> None:
        left = [
            ChecklistItem("IEC 61508_7.4.2", "7.4.2", "desc", "formal/", "partial"),
            ChecklistItem("IEC 61508_7.4.2_b", "7.4.2", "desc2", "formal/", "partial"),
        ]
        right = [ChecklistItem("ISO 26262_6.7.4", "6.7.4", "desc", "formal/", "partial")]
        assert CrossStandardMapper.coverage_overlap(left, right) == 1

    @pytest.mark.parametrize(
        ("standard", "clause", "match"), [("", "7.4.2", "standard"), ("IEC 61508", "", "clause")]
    )
    def test_equivalent_clauses_rejects_invalid_contracts(
        self, standard: Any, clause: Any, match: Any
    ) -> None:
        with pytest.raises(ValueError, match=match):
            CrossStandardMapper.equivalent_clauses(standard, clause)

    def test_equivalent_clauses_rejects_corrupted_mapping_state(self) -> None:
        key = ("IEC 61508", "7.4.2")
        original = CROSS_MAP[key]
        try:
            CROSS_MAP[key] = _unsafe([("ISO 26262", ""), ("DO-254", "6.0")])
            with pytest.raises(ValueError, match="mappings"):
                CrossStandardMapper.equivalent_clauses("IEC 61508", "7.4.2")
        finally:
            CROSS_MAP[key] = original

    @pytest.mark.parametrize(
        ("left", "right", "match"),
        [
            ("invalid", [], "lists"),
            ([], "invalid", "lists"),
            (["bad"], [], "checklist_a"),
            ([], ["bad"], "checklist_b"),
        ],
    )
    def test_coverage_overlap_rejects_invalid_contracts(
        self, left: Any, right: Any, match: Any
    ) -> None:
        with pytest.raises(ValueError, match=match):
            CrossStandardMapper.coverage_overlap(_unsafe(left), _unsafe(right))


class TestBoundaryContracts:
    def test_cross_standard_overlap_rejects_empty_clauses_on_each_side(self) -> None:
        left = ChecklistItem("IEC 61508_7.4.2", "7.4.2", "desc", "formal/", "partial")
        right = ChecklistItem("ISO 26262_6.7.4", "6.7.4", "desc", "formal/", "partial")
        left.clause = _unsafe("")
        with pytest.raises(ValueError, match="checklist_a clauses"):
            CrossStandardMapper.coverage_overlap([left], [right])
        left.clause = _unsafe("7.4.2")
        right.status = _unsafe("invalid")
        with pytest.raises(ValueError, match="checklist_b statuses"):
            CrossStandardMapper.coverage_overlap([left], [right])


class TestFailClosedEvidenceContracts:
    def test_addressed_item_requires_real_evidence(self) -> None:
        with pytest.raises(ValueError, match="require non-empty evidence"):
            ChecklistItem("id", "7.4.2", "description", "", "partial")

    def test_checklist_evidence_mapping_is_strictly_validated(self) -> None:
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
        with pytest.raises(ValueError, match="SILLevel"):
            IEC62304Assessment.from_sil(_unsafe("SIL_2"))

    def test_multiple_risk_controls_are_all_validated(self) -> None:
        assessment = IEC62304Assessment(
            SWClass.CLASS_B,
            risk_control_measures=["monitor", "independent shutdown"],
        )
        assert assessment.requires_unit_testing
