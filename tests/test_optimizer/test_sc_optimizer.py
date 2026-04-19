# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Optimizer Tests

"""Tests for the stochastic optimizer module."""

from __future__ import annotations

import unittest

from sc_neurocore.optimizer.sc_optimizer import (
    SCOptimizer,
    HardwareBudget,
    LayerProfile,
    LayerConfig,
)


class TestResourceEstimation(unittest.TestCase):
    def setUp(self):
        self.opt = SCOptimizer(HardwareBudget(max_luts=100000, max_power_mw=1000.0))

    def test_deterministic_mode_baseline(self):
        luts, power, accuracy, latency = self.opt._estimate_resources(
            100, 1, "None", "Deterministic"
        )
        self.assertEqual(accuracy, 1.0)
        self.assertGreater(luts, 0)
        self.assertGreater(power, 0)

    def test_sc_lfsr_cheaper_than_deterministic(self):
        det_luts, _, _, _ = self.opt._estimate_resources(100, 256, "None", "Deterministic")
        sc_luts, _, _, _ = self.opt._estimate_resources(100, 256, "LFSR", "SC")
        self.assertLess(sc_luts, det_luts)

    def test_sobol_more_accurate_than_lfsr(self):
        _, _, acc_lfsr, _ = self.opt._estimate_resources(100, 256, "LFSR", "SC")
        _, _, acc_sobol, _ = self.opt._estimate_resources(100, 256, "Sobol", "SC")
        self.assertGreater(acc_sobol, acc_lfsr)

    def test_longer_bitstream_more_accurate(self):
        _, _, acc_short, _ = self.opt._estimate_resources(100, 64, "LFSR", "SC")
        _, _, acc_long, _ = self.opt._estimate_resources(100, 1024, "LFSR", "SC")
        self.assertGreater(acc_long, acc_short)

    def test_accuracy_clamped(self):
        _, _, acc, _ = self.opt._estimate_resources(100, 4, "None", "SC")
        self.assertGreaterEqual(acc, 0.1)
        self.assertLessEqual(acc, 1.0)


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


class TestOptimization(unittest.TestCase):
    def test_single_layer(self):
        opt = SCOptimizer(HardwareBudget(max_luts=100000, max_power_mw=1000.0))
        network = [LayerProfile(id="L0", mac_count=100)]
        report = opt.optimize(network)
        self.assertIsNotNone(report)
        self.assertIn("L0", report.config)
        self.assertIsInstance(report.config["L0"], LayerConfig)

    def test_multi_layer(self):
        opt = SCOptimizer(HardwareBudget(max_luts=1000000, max_power_mw=5000.0))
        network = [
            LayerProfile(id=f"L{i}", mac_count=500, is_critical_path=(i == 0)) for i in range(10)
        ]
        report = opt.optimize(network)
        self.assertIsNotNone(report)
        self.assertEqual(len(report.config), 10)

    def test_critical_path_gets_priority(self):
        opt = SCOptimizer(HardwareBudget(max_luts=1000000, max_power_mw=50000.0))
        network = [
            LayerProfile(id="critical", mac_count=100, is_critical_path=True),
            LayerProfile(id="normal", mac_count=100, is_critical_path=False),
        ]
        report = opt.optimize(network)
        self.assertIsNotNone(report)
        self.assertGreaterEqual(
            report.config["critical"].accuracy_score,
            report.config["normal"].accuracy_score,
        )

    def test_tiny_budget_returns_none(self):
        opt = SCOptimizer(HardwareBudget(max_luts=1, max_power_mw=0.001))
        network = [LayerProfile(id="L0", mac_count=10000)]
        result = opt.optimize(network)
        self.assertIsNone(result)

    def test_resource_budget_respected(self):
        budget = HardwareBudget(max_luts=50000, max_power_mw=500.0)
        opt = SCOptimizer(budget)
        network = [LayerProfile(id=f"L{i}", mac_count=200) for i in range(5)]
        report = opt.optimize(network)
        if report:
            self.assertLessEqual(report.total_luts, budget.max_luts)
            self.assertLessEqual(report.total_power_mw, budget.max_power_mw)


if __name__ == "__main__":
    unittest.main()
