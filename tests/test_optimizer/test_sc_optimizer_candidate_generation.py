# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCandidateGeneration from former test_sc_optimizer.py

"""Focused suite: TestCandidateGeneration from former test_sc_optimizer.py."""

from __future__ import annotations

from sc_optimizer_support import *  # noqa: F403

class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        self.opt = SCOptimizer(HardwareBudget(max_luts=100000, max_power_mw=1000.0))

    def test_generates_candidates(self):
        layer = LayerProfile(id="L0", mac_count=100)
        candidates = self.opt._generate_candidates(layer)
        self.assertGreater(len(candidates), 0)
        modes = {c.mode for c in candidates}
        self.assertIn("SC", modes)
        self.assertIn("Deterministic", modes)

    def test_deterministic_candidate_exists(self):
        layer = LayerProfile(id="L0", mac_count=50)
        candidates = self.opt._generate_candidates(layer)
        det = [c for c in candidates if c.mode == "Deterministic"]
        self.assertEqual(len(det), 1)
