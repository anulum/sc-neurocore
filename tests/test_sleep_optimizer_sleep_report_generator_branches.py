# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepReportGeneratorBranches from former test_sleep_optimizer.py

"""Focused suite: TestSleepReportGeneratorBranches from former test_sleep_optimizer.py."""

from __future__ import annotations

from tests.sleep_optimizer_support import *  # noqa: F403


class TestSleepReportGeneratorBranches(unittest.TestCase):
    """Grade bands, sleep-onset scoring, wakeup counting, and recommendations.

    Builds a controlled tick history directly so the quality composite lands in a
    chosen band deterministically. The config uses ``sample_rate=1`` and
    ``stage_check_interval=60`` so each tick is exactly one minute — the
    sleep-onset latency in minutes equals the number of leading WAKE ticks.
    """

    @staticmethod
    def _make_optimizer(stages, matches, *, reinductions=0):
        cfg = SleepOptimizerConfig(sample_rate=1, fft_window=256, stage_check_interval=60)
        opt = SleepOptimizer("insomnia_relief", config=cfg)
        opt.start_session()
        opt._reinduction_count = reinductions
        for i, (stage, match) in enumerate(zip(stages, matches)):
            opt._history.append(
                SleepTick(
                    tick=i + 1,
                    current_stage=stage,
                    target_stage=(stage if match else SleepStage.WAKE),
                    stage_match=match,
                )
            )
        return opt

    def test_grade_a_for_perfect_session(self):
        opt = self._make_optimizer([SleepStage.N2] * 10, [True] * 10)
        r = SleepReportGenerator().generate(opt)
        self.assertEqual(r.grade, "A")

    def test_grade_b_for_half_matched_session(self):
        opt = self._make_optimizer([SleepStage.N2] * 10, [True] * 5 + [False] * 5)
        r = SleepReportGenerator().generate(opt)
        self.assertEqual(r.grade, "B")

    def test_grade_d_for_delayed_onset_unmatched(self):
        # 20 min of WAKE onset then ten unmatched sleep epochs.
        stages = [SleepStage.WAKE] * 20 + [SleepStage.N1] * 10
        opt = self._make_optimizer(stages, [False] * 30)
        r = SleepReportGenerator().generate(opt)
        self.assertEqual(r.grade, "D")

    def test_grade_f_for_all_wake_session(self):
        opt = self._make_optimizer([SleepStage.WAKE] * 40, [False] * 40)
        r = SleepReportGenerator().generate(opt)
        self.assertEqual(r.grade, "F")

    def test_wakeups_after_onset_counted_and_flagged(self):
        stages = [
            SleepStage.N2,
            SleepStage.WAKE,
            SleepStage.N2,
            SleepStage.WAKE,
            SleepStage.N2,
            SleepStage.WAKE,
            SleepStage.N2,
        ]
        opt = self._make_optimizer(stages, [True] * len(stages))
        r = SleepReportGenerator().generate(opt)
        self.assertEqual(r.wakeups, 3)
        self.assertTrue(any("awakening" in rec for rec in r.recommendations))

    def test_low_n3_and_rem_emit_recommendations(self):
        # An all-N2 session has zero N3/REM against a protocol that targets them.
        opt = self._make_optimizer([SleepStage.N2] * 10, [True] * 10)
        r = SleepReportGenerator().generate(opt)
        joined = " ".join(r.recommendations)
        self.assertIn("N3", joined)
        self.assertIn("REM", joined)
