# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrojanLint from former test_intelligence_security_and_compliance.py

"""Focused suite: TestTrojanLint from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

from tests.intelligence_security_and_compliance_support import *  # noqa: F403

class TestTrojanLint:
    def test_clean(self):
        from sc_neurocore.compiler.intelligence import lint_hardware_trojans

        r = lint_hardware_trojans({"v": "a + b", "u": "c - d"})
        assert r.risk_level == "LOW"

    def test_conditional(self):
        from sc_neurocore.compiler.intelligence import lint_hardware_trojans

        r = lint_hardware_trojans({"v": "a if trigger else b"})
        assert r.risk_level in ("MEDIUM", "HIGH")
        assert len(r.suspicious_paths) >= 1
