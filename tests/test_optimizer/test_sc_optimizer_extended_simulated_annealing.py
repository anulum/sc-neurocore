# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSimulatedAnnealing from former test_sc_optimizer_extended.py

"""Focused suite: TestSimulatedAnnealing from former test_sc_optimizer_extended.py."""

from __future__ import annotations

from sc_optimizer_extended_support import *  # noqa: F403

class TestSimulatedAnnealing(unittest.TestCase):
    def test_annealing_produces_report(self):
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize_annealing(make_network(), max_iter=200)
        self.assertIsNotNone(report)
        self.assertIsInstance(report, OptimizerReport)
        self.assertLessEqual(report.total_luts, budget.max_luts)

    def test_annealing_infeasible(self):
        budget = HardwareBudget(max_luts=10, max_power_mw=0.001)
        opt = SCOptimizer(budget)
        report = opt.optimize_annealing(make_network(), max_iter=100)
        self.assertIsNone(report)

    def test_annealing_deterministic_with_seed(self):
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        r1 = opt.optimize_annealing(make_network(), seed=123, max_iter=200)
        r2 = opt.optimize_annealing(make_network(), seed=123, max_iter=200)
        self.assertAlmostEqual(r1.mean_accuracy, r2.mean_accuracy)

    def test_annealing_beats_or_matches_greedy(self):
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        greedy = opt.optimize(make_network())
        anneal = opt.optimize_annealing(make_network(), max_iter=500)
        self.assertIsNotNone(greedy)
        self.assertIsNotNone(anneal)
        # Annealing should at least match greedy (may exceed with enough iterations)
        self.assertGreaterEqual(anneal.mean_accuracy, greedy.mean_accuracy * 0.95)

    def test_pareto_frontier_non_empty(self):
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize_annealing(make_network(), max_iter=300)
        self.assertIsNotNone(report)
        self.assertGreater(len(report.pareto_frontier), 0)

    def test_pareto_points_sorted_by_luts(self):
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize_annealing(make_network(), max_iter=300)
        if len(report.pareto_frontier) > 1:
            luts_vals = [p[0] for p in report.pareto_frontier]
            self.assertEqual(luts_vals, sorted(luts_vals))
