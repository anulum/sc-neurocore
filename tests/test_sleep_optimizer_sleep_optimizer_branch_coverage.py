# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepOptimizerBranchCoverage from former test_sleep_optimizer.py

"""Focused suite: TestSleepOptimizerBranchCoverage from former test_sleep_optimizer.py."""

from __future__ import annotations

from tests.sleep_optimizer_support import *  # noqa: F403

class TestSleepOptimizerBranchCoverage(unittest.TestCase):
    """Inactive-session guards, the None-stage fallback, and re-induction logic."""

    def setUp(self):
        self.cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)

    def test_add_sample_before_start_is_ignored(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.add_sample(0.1)  # session not started -> no-op
        self.assertEqual(opt._sample_count, 0)

    def test_add_samples_before_start_is_ignored(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.add_samples(np.zeros(16))  # session not started -> no-op
        self.assertEqual(opt._sample_count, 0)

    def test_check_and_adapt_inactive_returns_none(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        self.assertIsNone(opt.check_and_adapt())

    def test_none_detection_falls_back_to_wake(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        with mock.patch.object(opt._detector, "detect", return_value=None):
            opt._sample_count = self.cfg.stage_check_interval
            tick = opt.check_and_adapt()
        self.assertIsNotNone(tick)
        self.assertEqual(tick.current_stage, SleepStage.WAKE)

    def test_repeated_wake_against_sleep_target_triggers_reinduction(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        interval = self.cfg.stage_check_interval
        # A detected WAKE while the protocol targets a sleep stage is an unwanted
        # awakening; two in a row within the attempt budget arms re-induction.
        with (
            mock.patch.object(opt._detector, "detect", return_value=SleepStage.WAKE),
            mock.patch.object(opt.protocol, "get_target_stage", return_value=SleepStage.N2),
        ):
            opt._sample_count = interval
            first = opt.check_and_adapt()
            opt._sample_count = 2 * interval
            second = opt.check_and_adapt()
        self.assertEqual(first.current_stage, SleepStage.WAKE)
        self.assertFalse(first.reinduction_active)
        self.assertTrue(second.reinduction_active)
        self.assertEqual(opt._reinduction_count, 1)
