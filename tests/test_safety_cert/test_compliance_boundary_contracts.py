# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundaryContracts from former test_compliance.py

"""Focused suite: TestBoundaryContracts from former test_compliance.py."""

from __future__ import annotations

from tests.test_safety_cert.compliance_support import *  # noqa: F403

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
