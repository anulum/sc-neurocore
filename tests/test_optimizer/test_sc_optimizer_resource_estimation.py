# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResourceEstimation from former test_sc_optimizer.py

"""Focused suite: TestResourceEstimation from former test_sc_optimizer.py."""

from __future__ import annotations

from sc_optimizer_support import *  # noqa: F403


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
