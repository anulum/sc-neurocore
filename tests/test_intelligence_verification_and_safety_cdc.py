# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCDC from former test_intelligence_verification_and_safety.py

"""Focused suite: TestCDC from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403


class TestCDC:
    def test_same_domain(self):
        from sc_neurocore.compiler.intelligence import analyze_cdc

        r = analyze_cdc({"v": "a + b", "u": "c"})
        assert r.safe is True

    def test_cross_domain(self):
        from sc_neurocore.compiler.intelligence import analyze_cdc

        r = analyze_cdc(
            {"v": "u + 1", "u": "v - 1"},
            clock_domains={"v": "clk_a", "u": "clk_b"},
        )
        assert r.total_crossings >= 2
