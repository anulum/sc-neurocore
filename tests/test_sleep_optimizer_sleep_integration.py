# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepIntegration from former test_sleep_optimizer.py

"""Focused suite: TestSleepIntegration from former test_sleep_optimizer.py."""

from __future__ import annotations

from tests.sleep_optimizer_support import *  # noqa: F403

class TestSleepIntegration(unittest.TestCase):
    def test_full_flow(self):
        circ = CircadianOptimizer(Chronotype.DOLPHIN)
        proto_name = circ.get_recommended_protocol()
        cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)
        opt = SleepOptimizer(proto_name, config=cfg)
        opt.start_session()
        for i, stage in enumerate(
            [SleepStage.WAKE, SleepStage.N1, SleepStage.N2, SleepStage.N3, SleepStage.REM] * 5
        ):
            opt.add_samples(generate_stage_eeg(stage, n_samples=256, seed=i))
            opt.check_and_adapt()
        opt.stop_session()
        r = SleepReportGenerator().generate(opt)
        self.assertGreater(r.total_duration_min, 0)

    def test_all_protocols_run(self):
        cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)
        for name in PROTOCOL_REGISTRY:
            opt = SleepOptimizer(name, config=cfg)
            opt.start_session()
            for i in range(10):
                opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i))
                opt.check_and_adapt()
            self.assertEqual(len(opt.get_history()), 10)

    def test_session_reset(self):
        cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)
        opt = SleepOptimizer("insomnia_relief", config=cfg)
        opt.start_session()
        for i in range(5):
            opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i))
            opt.check_and_adapt()
        # Re-start clears history
        opt.start_session()
        self.assertEqual(len(opt.get_history()), 0)
