# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSideChannelLint from former test_intelligence_security_and_compliance.py

"""Focused suite: TestSideChannelLint from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

from tests.intelligence_security_and_compliance_support import *  # noqa: F403

class TestSideChannelLint:
    """Side-channel leakage analysis."""

    def test_clean_expression(self):
        from sc_neurocore.compiler.intelligence import (
            lint_side_channels,
        )

        findings = lint_side_channels({"v": "a + b"})
        # Should still have spike_out finding
        assert any(f.signal == "spike_out" for f in findings)

    def test_division_flagged(self):
        from sc_neurocore.compiler.intelligence import (
            lint_side_channels,
        )

        findings = lint_side_channels({"v": "a / b"})
        div_findings = [f for f in findings if "Division" in f.description]
        assert len(div_findings) == 1
        assert div_findings[0].risk_level == "medium"

    def test_branch_flagged(self):
        from sc_neurocore.compiler.intelligence import (
            lint_side_channels,
        )

        findings = lint_side_channels({"v": "a if x > 0 else b"})
        branch = [f for f in findings if f.risk_level == "high"]
        assert len(branch) >= 1

    def test_multiply_flagged(self):
        from sc_neurocore.compiler.intelligence import (
            lint_side_channels,
        )

        findings = lint_side_channels({"v": "a * b"})
        mul = [f for f in findings if "Hamming" in f.description]
        assert len(mul) == 1
