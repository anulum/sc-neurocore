# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossStandardMapper from former test_compliance.py

"""Focused suite: TestCrossStandardMapper from former test_compliance.py."""

from __future__ import annotations

from tests.test_safety_cert.compliance_support import *  # noqa: F403


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
