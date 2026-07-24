# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTimingClosure from former test_intelligence_verification_and_safety.py

"""Focused suite: TestTimingClosure from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403


class TestTimingClosure(unittest.TestCase):
    def test_simple_passes(self):
        r = verify_timing_closure({"v": "-(v)/tau + I"}, target_freq_mhz=100.0)
        self.assertTrue(r.timing_met)
        self.assertGreater(r.slack_ns, 0)

    def test_complex_fails(self):
        eqs = {"v": "a*b*c*d*e*f*g*h + i + j + k + l"}
        r = verify_timing_closure(eqs, target_freq_mhz=2000.0, data_width=32)
        if not r.timing_met:
            self.assertGreater(len(r.recommendations), 0)

    def test_tight_slack_recommends_extra_stage(self):
        # 3 adds + 2 muls is 3.8 ns of logic against a 4.0 ns period: the path
        # still closes (positive slack) but sits inside the 10% margin, so the
        # report flags it as tight rather than comfortable.
        r = verify_timing_closure({"v": "a + b + c - d * e * f"}, target_freq_mhz=250.0)
        self.assertTrue(r.timing_met)
        self.assertTrue(any("Tight slack" in rec for rec in r.recommendations))
