# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDecorrelationStrategies from former test_sc_optimizer_extended.py

"""Focused suite: TestDecorrelationStrategies from former test_sc_optimizer_extended.py."""

from __future__ import annotations

from sc_optimizer_extended_support import *  # noqa: F403


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
