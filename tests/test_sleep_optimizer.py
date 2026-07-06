# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Sleep Optimization System (UC3) — adapts to

"""Tests for Sleep Optimization System (UC3) — adapts to agent-written API."""

from __future__ import annotations
import unittest
from unittest import mock
import numpy as np

from sc_neurocore.sleep import (
    SleepStageDetector,
    SleepStage,
    CircadianOptimizer,
    Chronotype,
    get_protocol,
    list_protocols,
    SleepOptimizer,
    SleepOptimizerConfig,
    SleepReportGenerator,
    SleepReport,
)
from sc_neurocore.sleep.sleep_stage_detector import DetectorConfig, STAGE_SIGNATURES, EEG_BANDS
from sc_neurocore.sleep.protocol_library import StageAudioParams, PROTOCOL_REGISTRY


def generate_stage_eeg(stage, sample_rate=256, n_samples=512, seed=42):
    t = np.arange(n_samples) / sample_rate
    rng = np.random.RandomState(seed)
    if stage == SleepStage.WAKE:
        signal = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
    elif stage == SleepStage.N1:
        signal = 0.6 * np.sin(2 * np.pi * 6 * t) + 0.2 * np.sin(2 * np.pi * 10 * t)
    elif stage == SleepStage.N2:
        signal = 0.5 * np.sin(2 * np.pi * 3 * t) + 0.4 * np.sin(2 * np.pi * 1.5 * t)
    elif stage == SleepStage.N3:
        signal = 0.8 * np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.sin(2 * np.pi * 0.8 * t)
    elif stage == SleepStage.REM:
        signal = 0.4 * np.sin(2 * np.pi * 6 * t) + 0.3 * np.sin(2 * np.pi * 15 * t)
    else:
        signal = np.zeros(n_samples)
    return signal + rng.normal(0, 0.15, n_samples)


class TestSleepStageDetector(unittest.TestCase):
    def test_default_construction(self):
        det = SleepStageDetector()
        self.assertIsNotNone(det.config)

    def test_detect_without_enough_samples(self):
        det = SleepStageDetector(DetectorConfig(min_samples=128))
        for _ in range(50):
            det.add_sample(0.0)
        self.assertIsNone(det.detect())

    def test_add_samples_bulk(self):
        det = SleepStageDetector()
        det.add_samples(np.zeros(100))
        self.assertEqual(len(det._buffer), 100)

    def test_detect_wake(self):
        det = SleepStageDetector(DetectorConfig(sample_rate=256, fft_window=512))
        det.add_samples(generate_stage_eeg(SleepStage.WAKE, n_samples=512))
        result = det.detect()
        self.assertIsNotNone(result)
        self.assertIn(result, [SleepStage.WAKE, SleepStage.N1, SleepStage.REM])

    def test_detect_n3(self):
        det = SleepStageDetector(DetectorConfig(sample_rate=256, fft_window=512))
        det.add_samples(generate_stage_eeg(SleepStage.N3, n_samples=512))
        result = det.detect()
        self.assertIsNotNone(result)
        self.assertIn(result, [SleepStage.N3, SleepStage.N2])

    def test_band_powers_computed(self):
        det = SleepStageDetector(DetectorConfig(sample_rate=256, fft_window=256))
        det.add_samples(generate_stage_eeg(SleepStage.WAKE, n_samples=256))
        det.detect()
        powers = det.get_band_powers()
        for band in ["alpha", "beta", "delta", "theta", "gamma"]:
            self.assertIn(band, powers)
            self.assertGreaterEqual(powers[band], 0.0)

    def test_classify_static(self):
        # _classify is a static method taking a numpy power vector
        power_vec = np.array([0.05, 0.10, 0.35, 0.35, 0.15])  # WAKE signature
        stage = SleepStageDetector._classify(power_vec)
        self.assertEqual(stage, SleepStage.WAKE)

    def test_classify_n3(self):
        power_vec = np.array([0.60, 0.20, 0.10, 0.07, 0.03])  # N3 signature
        stage = SleepStageDetector._classify(power_vec)
        self.assertEqual(stage, SleepStage.N3)

    def test_reset(self):
        det = SleepStageDetector()
        det.add_samples(np.ones(100))
        det.detect()
        det.reset()
        self.assertEqual(len(det._buffer), 0)

    def test_stage_signatures_complete(self):
        for stage in SleepStage:
            self.assertIn(stage, STAGE_SIGNATURES)

    def test_eeg_bands_defined(self):
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            self.assertIn(band, EEG_BANDS)


class TestCircadianOptimizer(unittest.TestCase):
    def test_all_chronotypes_work(self):
        for ct in Chronotype:
            opt = CircadianOptimizer(ct)
            self.assertIsNotNone(opt.get_profile())

    def test_bear_defaults(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        p = opt.get_profile()
        self.assertEqual(p.bedtime_hour, 23.0)
        self.assertEqual(p.wake_hour, 7.0)

    def test_sleep_window(self):
        w = CircadianOptimizer(Chronotype.BEAR).get_sleep_window()
        self.assertIsInstance(w, tuple)
        self.assertEqual(len(w), 2)

    def test_is_in_sleep_window(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        self.assertTrue(opt.is_in_sleep_window(23.5))
        self.assertTrue(opt.is_in_sleep_window(2.0))
        self.assertFalse(opt.is_in_sleep_window(12.0))

    def test_recommended_protocol(self):
        for ct in Chronotype:
            proto = CircadianOptimizer(ct).get_recommended_protocol()
            self.assertIn(proto, PROTOCOL_REGISTRY)

    def test_melatonin_level(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        level = opt.melatonin_level(23.0)
        self.assertGreaterEqual(level, 0.0)
        self.assertLessEqual(level, 1.0)

    def test_melatonin_daytime_low(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        self.assertLess(opt.melatonin_level(14.0), 0.3)

    def test_to_dict(self):
        d = CircadianOptimizer(Chronotype.LION).to_dict()
        self.assertEqual(d["chronotype"], "lion")

    def test_different_protocols_for_different_types(self):
        wolf = CircadianOptimizer(Chronotype.WOLF).get_recommended_protocol()
        lion = CircadianOptimizer(Chronotype.LION).get_recommended_protocol()
        self.assertNotEqual(wolf, lion)


class TestProtocolLibrary(unittest.TestCase):
    def test_six_protocols(self):
        self.assertEqual(len(PROTOCOL_REGISTRY), 6)

    def test_get_valid(self):
        self.assertEqual(get_protocol("insomnia_relief").name, "insomnia_relief")

    def test_get_invalid(self):
        with self.assertRaises((ValueError, KeyError)):
            get_protocol("nonexistent")

    def test_list(self):
        protos = list_protocols()
        self.assertEqual(len(protos), 6)

    def test_stage_params(self):
        for name, proto in PROTOCOL_REGISTRY.items():
            for stage in SleepStage:
                audio = proto.get_audio_for_stage(stage)
                self.assertIsInstance(audio, StageAudioParams)

    def test_targets_sum_one(self):
        for name, proto in PROTOCOL_REGISTRY.items():
            total = sum(proto.stage_targets.values())
            self.assertAlmostEqual(total, 1.0, places=2, msg=f"{name} targets sum={total}")

    def test_target_stage_progression(self):
        proto = get_protocol("insomnia_relief")
        # get_target_stage takes a single float progress in [0, 1]
        early = proto.get_target_stage(0.01)
        self.assertIn(early, list(SleepStage))
        mid = proto.get_target_stage(0.5)
        self.assertIn(mid, list(SleepStage))

    def test_to_dict(self):
        d = get_protocol("power_nap").to_dict()
        self.assertEqual(d["name"], "power_nap")

    def test_power_nap_short(self):
        self.assertLessEqual(get_protocol("power_nap").total_duration_min, 30.0)


class TestSleepOptimizer(unittest.TestCase):
    def setUp(self):
        self.cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)

    def test_start(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()

    def test_protocol_from_string(self):
        opt = SleepOptimizer("deep_sleep_boost", config=self.cfg)
        opt.start_session()
        self.assertEqual(opt.protocol.name, "deep_sleep_boost")

    def test_check_none_without_samples(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_sample(0.1)
        self.assertIsNone(opt.check_and_adapt())

    def test_check_with_samples(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256))
        result = opt.check_and_adapt()
        self.assertIsNotNone(result)

    def test_tick_increments(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        for i in range(3):
            opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i))
            r = opt.check_and_adapt()
        self.assertEqual(r.tick, 3)

    def test_audio_params(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256))
        r = opt.check_and_adapt()
        # audio_params is a StageAudioParams dataclass
        self.assertIsInstance(r.audio_params, StageAudioParams)
        self.assertGreater(r.audio_params.binaural_hz, 0)

    def test_stage_durations(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        for i in range(5):
            opt.add_samples(generate_stage_eeg(SleepStage.N3, n_samples=256, seed=i))
            opt.check_and_adapt()
        self.assertGreater(sum(opt.get_stage_durations().values()), 0)

    def test_history(self):
        opt = SleepOptimizer("deep_sleep_boost", config=self.cfg)
        opt.start_session()
        for i in range(3):
            opt.add_samples(generate_stage_eeg(SleepStage.N3, n_samples=256, seed=i))
            opt.check_and_adapt()
        self.assertEqual(len(opt.get_history()), 3)

    def test_hypnogram(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        for i in range(5):
            opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i))
            opt.check_and_adapt()
        self.assertGreater(len(opt.get_hypnogram()), 0)

    def test_stop(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        result = opt.stop_session()
        self.assertIsInstance(result, list)

    def test_state(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.REM, n_samples=256))
        opt.check_and_adapt()
        s = opt.get_state()
        self.assertIsInstance(s, dict)
        self.assertIn("protocol", s)

    def test_tick_has_current_stage(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256))
        tick = opt.check_and_adapt()
        self.assertIn(tick.current_stage, list(SleepStage))


class TestSleepReportGenerator(unittest.TestCase):
    def _run_session(self, protocol="insomnia_relief", n_epochs=50):
        cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)
        opt = SleepOptimizer(protocol, config=cfg)
        opt.start_session()
        stages = [
            SleepStage.WAKE,
            SleepStage.N1,
            SleepStage.N2,
            SleepStage.N3,
            SleepStage.N2,
            SleepStage.REM,
            SleepStage.N1,
        ]
        for i in range(n_epochs):
            stage = stages[min(int(i / n_epochs * len(stages)), len(stages) - 1)]
            opt.add_samples(generate_stage_eeg(stage, n_samples=256, seed=i))
            opt.check_and_adapt()
        opt.stop_session()
        return opt

    def test_generate(self):
        r = SleepReportGenerator().generate(self._run_session())
        self.assertIsInstance(r, SleepReport)
        self.assertGreater(r.total_duration_min, 0)

    def test_quality_range(self):
        r = SleepReportGenerator().generate(self._run_session())
        self.assertGreaterEqual(r.quality_score, 0)
        self.assertLessEqual(r.quality_score, 100)

    def test_grade(self):
        r = SleepReportGenerator().generate(self._run_session())
        self.assertIn(r.grade, "ABCDF")

    def test_efficiency(self):
        r = SleepReportGenerator().generate(self._run_session())
        self.assertGreaterEqual(r.sleep_efficiency_pct, 0)
        self.assertLessEqual(r.sleep_efficiency_pct, 100)

    def test_empty_session(self):
        cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)
        opt = SleepOptimizer("insomnia_relief", config=cfg)
        opt.start_session()
        opt.stop_session()
        r = SleepReportGenerator().generate(opt)
        self.assertEqual(r.total_duration_min, 0.0)


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


if __name__ == "__main__":
    unittest.main()
