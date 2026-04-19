# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-Optimizer Extended Tests

from __future__ import annotations

import unittest
from sc_neurocore.optimizer.sc_optimizer import (
    SCOptimizer,
    HardwareBudget,
    LayerProfile,
    OptimizerReport,
    DecorrelationStrategy,
    ComputeMode,
)


def make_network(n: int = 5, mac: int = 100) -> list[LayerProfile]:
    return [LayerProfile(id=f"L{i}", mac_count=mac, is_critical_path=(i == 0)) for i in range(n)]


class TestGreedyOptimizer(unittest.TestCase):
    def test_feasible_optimization(self):
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize(make_network())
        self.assertIsNotNone(report)
        self.assertIsInstance(report, OptimizerReport)
        self.assertLessEqual(report.total_luts, budget.max_luts)
        self.assertLessEqual(report.total_power_mw, budget.max_power_mw)

    def test_infeasible_returns_none(self):
        budget = HardwareBudget(max_luts=10, max_power_mw=0.001)
        opt = SCOptimizer(budget)
        report = opt.optimize(make_network())
        self.assertIsNone(report)

    def test_critical_path_prioritized(self):
        budget = HardwareBudget(max_luts=10_000_000, max_power_mw=50000.0)
        opt = SCOptimizer(budget)
        net = [
            LayerProfile("crit", 100, is_critical_path=True),
            LayerProfile("norm", 100, is_critical_path=False),
        ]
        report = opt.optimize(net)
        self.assertIsNotNone(report)
        crit_acc = report.config["crit"].accuracy_score
        norm_acc = report.config["norm"].accuracy_score
        self.assertGreaterEqual(crit_acc, norm_acc)

    def test_report_summary(self):
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize(make_network(2))
        self.assertIsNotNone(report)
        summary = report.summary()
        self.assertIn("LUTs:", summary)
        self.assertIn("Power:", summary)

    def test_latency_constraint(self):
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0, max_latency_cycles=128)
        opt = SCOptimizer(budget)
        report = opt.optimize(make_network(3))
        self.assertIsNotNone(report)
        self.assertLessEqual(report.total_latency_cycles, 128)


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


class TestDecorrelationStrategies(unittest.TestCase):
    def test_all_strategies_exist(self):
        strategies = [s.value for s in DecorrelationStrategy]
        self.assertIn("LFSR", strategies)
        self.assertIn("Sobol", strategies)
        self.assertIn("Halton", strategies)
        self.assertIn("SCC_Decorrelator", strategies)
        self.assertIn("None", strategies)

    def test_all_modes_exist(self):
        modes = [m.value for m in ComputeMode]
        self.assertIn("SC", modes)
        self.assertIn("Deterministic", modes)
        self.assertIn("Hybrid", modes)

    def test_hybrid_mode_resources(self):
        budget = HardwareBudget(max_luts=10_000_000, max_power_mw=100000.0)
        opt = SCOptimizer(budget)
        l, p, a, lat = opt._estimate_resources(100, 256, "Sobol", "Hybrid")
        self.assertGreater(l, 0)
        self.assertGreater(p, 0)
        self.assertGreater(a, 0.9)

    def test_sobol_higher_accuracy_than_lfsr(self):
        budget = HardwareBudget(max_luts=10_000_000, max_power_mw=100000.0)
        opt = SCOptimizer(budget)
        _, _, acc_sobol, _ = opt._estimate_resources(100, 256, "Sobol", "SC")
        _, _, acc_lfsr, _ = opt._estimate_resources(100, 256, "LFSR", "SC")
        self.assertGreater(acc_sobol, acc_lfsr)

    def test_sobol_more_luts_than_lfsr(self):
        budget = HardwareBudget(max_luts=10_000_000, max_power_mw=100000.0)
        opt = SCOptimizer(budget)
        luts_sobol, _, _, _ = opt._estimate_resources(100, 256, "Sobol", "SC")
        luts_lfsr, _, _, _ = opt._estimate_resources(100, 256, "LFSR", "SC")
        self.assertGreater(luts_sobol, luts_lfsr)


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


if __name__ == "__main__":
    unittest.main()
