# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestComplianceMatrix from former test_intelligence_verification_and_safety.py

"""Focused suite: TestComplianceMatrix from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403

class TestComplianceMatrix:
    """Safety compliance matrix generation."""

    def test_default_standards(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compliance_matrix,
        )

        entries = generate_compliance_matrix("sc_lif")
        standards = {e.standard for e in entries}
        assert "DO-254" in standards
        assert "IEC 61508" in standards
        assert "ISO 26262" in standards

    def test_all_covered(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compliance_matrix,
        )

        entries = generate_compliance_matrix(
            "sc_lif",
            has_tmr=True,
            has_checksum=True,
            has_sva=True,
            has_provenance=True,
        )
        covered = [e for e in entries if e.status == "covered"]
        assert len(covered) == len(entries)

    def test_gaps_without_tmr(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compliance_matrix,
        )

        entries = generate_compliance_matrix("sc_lif")
        gaps = [e for e in entries if e.status == "gap"]
        assert len(gaps) > 0

    def test_format_report(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compliance_matrix,
            format_compliance_report,
        )

        entries = generate_compliance_matrix("sc_lif", has_tmr=True)
        report = format_compliance_report(entries)
        assert "Compliance Matrix" in report
        assert "DO-254" in report
        assert "✅" in report
