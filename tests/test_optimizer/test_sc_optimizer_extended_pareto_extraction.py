# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParetoExtraction from former test_sc_optimizer_extended.py

"""Focused suite: TestParetoExtraction from former test_sc_optimizer_extended.py."""

from __future__ import annotations

from sc_optimizer_extended_support import *  # noqa: F403

class TestParetoExtraction(unittest.TestCase):
    def test_empty_input(self):
        result = SCOptimizer._extract_pareto([])
        self.assertEqual(result, [])

    def test_single_point(self):
        result = SCOptimizer._extract_pareto([(100, 1.0, 0.9)])
        self.assertEqual(len(result), 1)

    def test_dominated_removed(self):
        points = [
            (100, 1.0, 0.9),  # dominated by below
            (90, 0.8, 0.95),  # non-dominated
        ]
        result = SCOptimizer._extract_pareto(points)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 90)

    def test_non_dominated_kept(self):
        points = [
            (100, 1.0, 0.95),  # good accuracy, more resources
            (50, 0.5, 0.80),  # less resources, lower accuracy
        ]
        result = SCOptimizer._extract_pareto(points)
        self.assertEqual(len(result), 2)
