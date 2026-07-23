# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultCampaign from former test_intelligence_verification_and_safety.py

"""Focused suite: TestFaultCampaign from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403

class TestFaultCampaign(unittest.TestCase):
    def test_basic(self):
        r = run_fault_campaign({"v": "a", "u": "b"})
        self.assertEqual(r.total_injections, 1000)
        self.assertGreater(r.sdc_count, 0)
        self.assertGreater(r.sdc_rate, 0)
        self.assertLess(r.sdc_rate, 1.0)

    def test_deterministic(self):
        r1 = run_fault_campaign({"v": "a"}, seed=123)
        r2 = run_fault_campaign({"v": "a"}, seed=123)
        self.assertEqual(r1.sdc_count, r2.sdc_count)
