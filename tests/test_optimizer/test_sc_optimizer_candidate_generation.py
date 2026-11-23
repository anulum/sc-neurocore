# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCandidateGeneration from former test_sc_optimizer.py

"""Focused suite: TestCandidateGeneration from former test_sc_optimizer.py."""

from __future__ import annotations

import unittest
from sc_neurocore.optimizer.sc_optimizer import (
    HardwareBudget,
    LayerProfile,
    SCOptimizer,
)


class TestCandidateGeneration(unittest.TestCase):
    """Candidate configurations generated for a layer."""

    def setUp(self) -> None:
        """Create an optimizer with a generous budget."""
        self.opt = SCOptimizer(HardwareBudget(max_luts=100000, max_power_mw=1000.0))

    def test_generates_candidates(self) -> None:
        """Candidates cover both stochastic and deterministic modes."""
        layer = LayerProfile(id="L0", mac_count=100)
        candidates = self.opt._generate_candidates(layer)
        self.assertGreater(len(candidates), 0)
        modes = {c.mode for c in candidates}
        self.assertIn("SC", modes)
        self.assertIn("Deterministic", modes)

    def test_deterministic_candidate_exists(self) -> None:
        """Exactly one candidate is deterministic."""
        layer = LayerProfile(id="L0", mac_count=50)
        candidates = self.opt._generate_candidates(layer)
        det = [c for c in candidates if c.mode == "Deterministic"]
        self.assertEqual(len(det), 1)
