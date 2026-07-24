# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrojanLintBranches from former test_intelligence_security_and_compliance.py

"""Focused suite: TestTrojanLintBranches from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

from tests.intelligence_security_and_compliance_support import *  # noqa: F403


class TestTrojanLintBranches:
    """Cover the payload cross-reference accounting and the high-risk verdict
    that the single-equation lint cases never reach."""

    def test_payload_cross_reference_counts_as_check(self):
        from sc_neurocore.compiler.intelligence import lint_hardware_trojans

        # "u" appears inside v's expression, so a payload cross-reference check
        # runs on top of the two per-variable checks.
        r = lint_hardware_trojans({"v": "u + 1", "u": "2"})
        assert r.total_checks >= 3

    def test_two_conditional_paths_are_high_risk(self):
        from sc_neurocore.compiler.intelligence import lint_hardware_trojans

        r = lint_hardware_trojans({"a": "if x then 1", "b": "y ? 1 : 0"})
        assert r.risk_level == "HIGH"
        assert len(r.suspicious_paths) >= 2
