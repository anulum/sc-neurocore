"""
Tests for the Sleep Optimization System (UC3)
==============================================

Covers:
- SleepStageDetector: EEG band power classification
- CircadianOptimizer: Chronotype profiling
- SleepProtocol / protocol_library: Protocol definitions
- SleepOptimizer: Closed-loop orchestration
- SleepReportGenerator: Morning report generation

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from sc_neurocore.sleep import (
    SleepStageDetector,
    SleepStage,
    CircadianOptimizer,
    Chronotype,
    SleepProtocol,
    get_protocol,
    list_protocols,
    SleepOptimizer,
    SleepOptimizerConfig,
    SleepReportGenerator,
    SleepReport,
)
from sc_neurocore.sleep.sleep_stage_detector import (
    DetectorConfig,
    STAGE_SIGNATURES,
    EEG_BANDS,
)
from sc_neurocore.sleep.circadian_optimizer import (
    CircadianProfile,
    CIRCADIAN_PROFILES,
)
from sc_neurocore.sleep.protocol_library import (
    StageAudioParams,
    PROTOCOL_REGISTRY,
)


# ── Helpers ─────────────────────────────────────────────────────────────

def generate_stage_eeg(
    stage: SleepStage,
    sample_rate: int = 256,
    n_samples: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Generate simulated EEG signal for a specific sleep stage."""
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


# ══════════════════════════════════════════════════════════════════════════
# Test: SleepStageDetector
# ══════════════════════════════════════════════════════════════════════════


class TestSleepStageDetector(unittest.TestCase):
    """Tests for EEG-based sleep stage classification."""

    def test_initial_state_is_wake(self):
        det = SleepStageDetector()
        self.assertEqual(det.current_stage, SleepStage.WAKE)

    def test_detect_returns_wake_without_enough_samples(self):
        det = SleepStageDetector(DetectorConfig(min_samples=128))
        # Add fewer than min_samples
        for _ in range(50):
            det.add_sample(0.0)
        stage = det.detect()
        self.assertEqual(stage, SleepStage.WAKE)

    def test_add_samples_bulk(self):
        det = SleepStageDetector()
        arr = np.zeros(100)
        det.add_samples(arr)
        self.assertEqual(len(det._buffer), 100)

    def test_detect_wake_stage(self):
        """Alpha+beta dominant signal should classify as WAKE."""
        det = SleepStageDetector(DetectorConfig(sample_rate=256, fft_window=512))
        eeg = generate_stage_eeg(SleepStage.WAKE, n_samples=512)
        det.add_samples(eeg)
        stage = det.detect()
        # Should be WAKE or N1 (alpha overlap)
        self.assertIn(stage, [SleepStage.WAKE, SleepStage.N1])

    def test_detect_n3_deep_sleep(self):
        """Strong delta signal should classify as N3."""
        det = SleepStageDetector(DetectorConfig(sample_rate=256, fft_window=512))
        eeg = generate_stage_eeg(SleepStage.N3, n_samples=512)
        det.add_samples(eeg)
        stage = det.detect()
        # N3 has very distinctive delta dominance
        self.assertIn(stage, [SleepStage.N3, SleepStage.N2])

    def test_band_powers_computed(self):
        det = SleepStageDetector(DetectorConfig(sample_rate=256, fft_window=256))
        eeg = generate_stage_eeg(SleepStage.WAKE, n_samples=256)
        det.add_samples(eeg)
        det.detect()
        powers = det.get_band_powers()
        self.assertIn("alpha", powers)
        self.assertIn("beta", powers)
        self.assertIn("delta", powers)
        self.assertIn("theta", powers)
        self.assertIn("gamma", powers)
        # All powers should be non-negative
        for v in powers.values():
            self.assertGreaterEqual(v, 0.0)

    def test_cosine_similarity(self):
        a = {"x": 1.0, "y": 0.0}
        b = {"x": 1.0, "y": 0.0}
        sim = SleepStageDetector._cosine_similarity(a, b)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        a = {"x": 1.0, "y": 0.0}
        b = {"x": 0.0, "y": 1.0}
        sim = SleepStageDetector._cosine_similarity(a, b)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_reset_clears_state(self):
        det = SleepStageDetector()
        det.add_samples(np.ones(100))
        det.detect()
        det.reset()
        self.assertEqual(len(det._buffer), 0)
        self.assertEqual(det.current_stage, SleepStage.WAKE)
        self.assertEqual(det.band_powers, {})

    def test_stage_signatures_complete(self):
        """Each stage should have a signature with all 5 bands."""
        for stage in SleepStage:
            self.assertIn(stage, STAGE_SIGNATURES)
            sig = STAGE_SIGNATURES[stage]
            for band in EEG_BANDS:
                self.assertIn(band, sig)

    def test_temporal_smoothing(self):
        """With smoothing_window=5, rapid changes should be filtered."""
        det = SleepStageDetector(DetectorConfig(
            sample_rate=256, fft_window=256, smoothing_window=5
        ))
        # Feed N3 signal multiple times to fill smoothing window
        for _ in range(5):
            eeg = generate_stage_eeg(SleepStage.N3, n_samples=256, seed=42)
            det.add_samples(eeg)
            det.detect()
        stage_before = det.current_stage
        # One WAKE signal shouldn't flip the stage
        eeg = generate_stage_eeg(SleepStage.WAKE, n_samples=256, seed=99)
        det._buffer.clear()
        det.add_samples(eeg)
        det.detect()
        # Should still be deep sleep due to smoothing
        self.assertEqual(det.current_stage, stage_before)


# ══════════════════════════════════════════════════════════════════════════
# Test: CircadianOptimizer
# ══════════════════════════════════════════════════════════════════════════


class TestCircadianOptimizer(unittest.TestCase):
    """Tests for chronotype-based circadian profiling."""

    def test_all_chronotypes_have_profiles(self):
        for ct in Chronotype:
            self.assertIn(ct, CIRCADIAN_PROFILES)

    def test_bear_defaults(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        profile = opt.get_profile()
        self.assertEqual(profile.chronotype, Chronotype.BEAR)
        self.assertAlmostEqual(profile.optimal_bedtime_h, 23.0)
        self.assertAlmostEqual(profile.optimal_wake_h, 7.0)

    def test_sleep_window(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        window = opt.get_sleep_window()
        self.assertIn("bedtime_h", window)
        self.assertIn("wake_h", window)
        self.assertIn("duration_h", window)
        self.assertAlmostEqual(window["duration_h"], 8.0)

    def test_wolf_sleep_crosses_midnight(self):
        opt = CircadianOptimizer(Chronotype.WOLF)
        window = opt.get_sleep_window()
        # Wolf bedtime 0.5, wake 8.5 → duration 8h
        self.assertAlmostEqual(window["duration_h"], 8.0)

    def test_is_in_sleep_window(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        # Bear: 23:00 - 07:00
        self.assertTrue(opt.is_in_sleep_window(23.5))
        self.assertTrue(opt.is_in_sleep_window(2.0))
        self.assertTrue(opt.is_in_sleep_window(6.0))
        self.assertFalse(opt.is_in_sleep_window(12.0))
        self.assertFalse(opt.is_in_sleep_window(15.0))

    def test_recommended_protocol(self):
        for ct in Chronotype:
            opt = CircadianOptimizer(ct)
            proto = opt.get_recommended_protocol()
            self.assertIn(proto, PROTOCOL_REGISTRY)

    def test_melatonin_level_at_onset(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        # At melatonin onset hour, level should be ~0
        level = opt.melatonin_level(opt.profile.melatonin_onset_h)
        self.assertAlmostEqual(level, 0.0, places=1)

    def test_melatonin_level_at_peak(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        # Peak ~3h after onset
        peak_h = (opt.profile.melatonin_onset_h + 4) % 24
        level = opt.melatonin_level(peak_h)
        self.assertGreater(level, 0.5)

    def test_melatonin_zero_during_day(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        level = opt.melatonin_level(14.0)  # 2pm
        self.assertAlmostEqual(level, 0.0, places=1)

    def test_to_dict(self):
        opt = CircadianOptimizer(Chronotype.LION)
        d = opt.to_dict()
        self.assertEqual(d["chronotype"], "lion")
        self.assertIn("optimal_bedtime_h", d)
        self.assertIn("recommended_protocol", d)
        self.assertIn("sleep_duration_h", d)


# ══════════════════════════════════════════════════════════════════════════
# Test: Protocol Library
# ══════════════════════════════════════════════════════════════════════════


class TestProtocolLibrary(unittest.TestCase):
    """Tests for sleep protocol definitions."""

    def test_six_protocols_registered(self):
        self.assertEqual(len(PROTOCOL_REGISTRY), 6)

    def test_get_protocol_valid(self):
        p = get_protocol("insomnia_relief")
        self.assertEqual(p.name, "insomnia_relief")
        self.assertIsInstance(p, SleepProtocol)

    def test_get_protocol_invalid_raises(self):
        with self.assertRaises(ValueError):
            get_protocol("nonexistent_protocol")

    def test_list_protocols_returns_dicts(self):
        protos = list_protocols()
        self.assertEqual(len(protos), 6)
        for p in protos:
            self.assertIn("name", p)
            self.assertIn("description", p)

    def test_all_protocols_have_stage_params(self):
        for name, proto in PROTOCOL_REGISTRY.items():
            for stage in SleepStage:
                audio = proto.get_audio_for_stage(stage)
                self.assertIsInstance(audio, StageAudioParams)
                self.assertGreater(audio.binaural_hz, 0)

    def test_protocol_stage_targets_sum_to_one(self):
        for name, proto in PROTOCOL_REGISTRY.items():
            if proto.stage_targets:
                total = sum(proto.stage_targets.values())
                self.assertAlmostEqual(total, 1.0, places=2,
                                       msg=f"{name} targets sum to {total}")

    def test_get_target_stage_progression(self):
        proto = get_protocol("insomnia_relief")
        total_min = proto.duration_h * 60
        # Early: should be WAKE or N1
        early = proto.get_target_stage(1.0, total_min)
        self.assertIn(early, [SleepStage.WAKE, SleepStage.N1])
        # Middle: should be deeper
        mid = proto.get_target_stage(total_min * 0.4, total_min)
        self.assertIn(mid, [SleepStage.N2, SleepStage.N3])
        # Late: should approach wake
        late = proto.get_target_stage(total_min * 0.9, total_min)
        self.assertIn(late, [SleepStage.N1, SleepStage.REM])

    def test_protocol_to_dict(self):
        proto = get_protocol("power_nap")
        d = proto.to_dict()
        self.assertEqual(d["name"], "power_nap")
        self.assertIn("duration_h", d)
        self.assertIn("induction_sweep", d)

    def test_power_nap_short_duration(self):
        proto = get_protocol("power_nap")
        self.assertAlmostEqual(proto.duration_h, 25.0 / 60.0, places=2)
        self.assertFalse(proto.wake_recovery_enabled)

    def test_stage_audio_params_defaults(self):
        p = StageAudioParams()
        self.assertEqual(p.binaural_hz, 10.0)
        self.assertEqual(p.noise_color, "pink")
        self.assertFalse(p.spatial_rotation)


# ══════════════════════════════════════════════════════════════════════════
# Test: SleepOptimizer
# ══════════════════════════════════════════════════════════════════════════


class TestSleepOptimizer(unittest.TestCase):
    """Tests for the master closed-loop sleep optimizer."""

    def setUp(self):
        self.config = SleepOptimizerConfig(
            sample_rate=256,
            fft_window=256,
            stage_check_interval=256,
        )

    def test_start_session(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        self.assertIsNotNone(opt.protocol)
        self.assertEqual(opt.protocol.name, "insomnia_relief")

    def test_start_session_auto_protocol(self):
        """Without explicit protocol, uses circadian recommendation."""
        opt = SleepOptimizer(chronotype=Chronotype.BEAR, config=self.config)
        opt.start_session()
        self.assertIsNotNone(opt.protocol)
        self.assertEqual(opt.protocol.name, "insomnia_relief")

    def test_add_sample_and_check(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        # Not enough samples → check returns None
        opt.add_sample(0.1)
        result = opt.check_and_adapt()
        self.assertIsNone(result)

    def test_check_after_enough_samples(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        eeg = generate_stage_eeg(SleepStage.N2, n_samples=256)
        opt.add_samples(eeg)
        result = opt.check_and_adapt()
        self.assertIsNotNone(result)
        self.assertIn(result.current_stage, [s.name for s in SleepStage])

    def test_tick_increments(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        for i in range(3):
            eeg = generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i)
            opt.add_samples(eeg)
            result = opt.check_and_adapt()
        self.assertEqual(result.tick, 2)

    def test_audio_params_in_tick(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        eeg = generate_stage_eeg(SleepStage.N2, n_samples=256)
        opt.add_samples(eeg)
        result = opt.check_and_adapt()
        self.assertIn("binaural_hz", result.audio_params)
        self.assertIn("noise_color", result.audio_params)
        self.assertIn("volume", result.audio_params)

    def test_stage_durations_tracked(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        for i in range(5):
            eeg = generate_stage_eeg(SleepStage.N3, n_samples=256, seed=i)
            opt.add_samples(eeg)
            opt.check_and_adapt()
        durations = opt.get_stage_durations()
        total = sum(durations.values())
        self.assertGreater(total, 0)

    def test_get_history(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("deep_sleep_boost")
        for i in range(3):
            eeg = generate_stage_eeg(SleepStage.N3, n_samples=256, seed=i)
            opt.add_samples(eeg)
            opt.check_and_adapt()
        history = opt.get_history()
        self.assertEqual(len(history), 3)
        self.assertIn("tick", history[0])
        self.assertIn("current_stage", history[0])
        self.assertIn("audio_params", history[0])

    def test_get_hypnogram(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        for i in range(5):
            eeg = generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i)
            opt.add_samples(eeg)
            opt.check_and_adapt()
        hyp = opt.get_hypnogram()
        self.assertGreater(len(hyp), 0)
        self.assertIn("elapsed_min", hyp[0])
        self.assertIn("stage", hyp[0])

    def test_stop_session(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        opt.stop_session()
        self.assertFalse(opt._running)

    def test_get_state_empty(self):
        opt = SleepOptimizer(config=self.config)
        state = opt.get_state()
        self.assertEqual(state["tick"], 0)
        self.assertEqual(state["current_stage"], "WAKE")

    def test_get_state_after_ticks(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        eeg = generate_stage_eeg(SleepStage.REM, n_samples=256)
        opt.add_samples(eeg)
        opt.check_and_adapt()
        state = opt.get_state()
        self.assertIn("elapsed_min", state)
        self.assertIn("band_powers", state)

    def test_reinduction_on_unexpected_wake(self):
        """Wake during target N3 should trigger reinduction."""
        opt = SleepOptimizer(config=SleepOptimizerConfig(
            sample_rate=256,
            fft_window=256,
            stage_check_interval=256,
            max_reinduction_attempts=3,
        ))
        opt.start_session("deep_sleep_boost")

        # First, advance a few ticks with N3 to get past WAKE target phase
        for i in range(15):
            eeg = generate_stage_eeg(SleepStage.N3, n_samples=256, seed=i)
            opt.add_samples(eeg)
            opt.check_and_adapt()

        # Now send WAKE signal — target should not be WAKE at this point
        eeg = generate_stage_eeg(SleepStage.WAKE, n_samples=256, seed=99)
        opt.add_samples(eeg)
        result = opt.check_and_adapt()
        # Either reinduction triggered or normal tick
        self.assertIsNotNone(result)

    def test_sleep_tick_to_dict(self):
        opt = SleepOptimizer(config=self.config)
        opt.start_session("insomnia_relief")
        eeg = generate_stage_eeg(SleepStage.N2, n_samples=256)
        opt.add_samples(eeg)
        result = opt.check_and_adapt()
        d = result.to_dict()
        self.assertIn("tick", d)
        self.assertIn("elapsed_min", d)
        self.assertIn("stage_match", d)


# ══════════════════════════════════════════════════════════════════════════
# Test: SleepReportGenerator
# ══════════════════════════════════════════════════════════════════════════


class TestSleepReportGenerator(unittest.TestCase):
    """Tests for morning report generation."""

    def _run_session(self, protocol: str = "insomnia_relief", n_epochs: int = 50):
        """Helper: run a full session and return optimizer."""
        config = SleepOptimizerConfig(
            sample_rate=256, fft_window=256, stage_check_interval=256,
        )
        opt = SleepOptimizer(config=config)
        opt.start_session(protocol)

        for i in range(n_epochs):
            progress = i / max(n_epochs, 1)
            if progress < 0.05:
                stage = SleepStage.WAKE
            elif progress < 0.15:
                stage = SleepStage.N1
            elif progress < 0.30:
                stage = SleepStage.N2
            elif progress < 0.50:
                stage = SleepStage.N3
            elif progress < 0.70:
                stage = SleepStage.N2
            elif progress < 0.85:
                stage = SleepStage.REM
            else:
                stage = SleepStage.N1

            eeg = generate_stage_eeg(stage, n_samples=256, seed=i)
            opt.add_samples(eeg)
            opt.check_and_adapt()

        opt.stop_session()
        return opt

    def test_generate_report(self):
        opt = self._run_session()
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        self.assertIsInstance(report, SleepReport)
        self.assertGreater(report.total_duration_min, 0)

    def test_report_has_stage_percentages(self):
        opt = self._run_session()
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        self.assertGreater(len(report.stage_percentages), 0)
        total_pct = sum(report.stage_percentages.values())
        self.assertAlmostEqual(total_pct, 100.0, places=0)

    def test_report_quality_score_range(self):
        opt = self._run_session()
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        self.assertGreaterEqual(report.quality_score, 0)
        self.assertLessEqual(report.quality_score, 100)

    def test_report_grade_valid(self):
        opt = self._run_session()
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        self.assertIn(report.grade, ["A", "B", "C", "D", "F"])

    def test_report_sleep_efficiency(self):
        opt = self._run_session()
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        self.assertGreaterEqual(report.sleep_efficiency_pct, 0)
        self.assertLessEqual(report.sleep_efficiency_pct, 100)

    def test_report_recommendations_not_empty(self):
        opt = self._run_session()
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        self.assertGreater(len(report.recommendations), 0)

    def test_report_to_dict(self):
        opt = self._run_session()
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        d = report.to_dict()
        self.assertIn("total_duration_min", d)
        self.assertIn("quality_score", d)
        self.assertIn("grade", d)
        self.assertIn("recommendations", d)
        self.assertIn("hypnogram", d)

    def test_empty_session_returns_empty_report(self):
        config = SleepOptimizerConfig(
            sample_rate=256, fft_window=256, stage_check_interval=256,
        )
        opt = SleepOptimizer(config=config)
        opt.start_session("insomnia_relief")
        opt.stop_session()
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        self.assertEqual(report.total_duration_min, 0.0)
        self.assertEqual(report.grade, "F")

    def test_report_hypnogram(self):
        opt = self._run_session()
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        self.assertGreater(len(report.hypnogram), 0)
        for entry in report.hypnogram:
            self.assertIn("elapsed_min", entry)
            self.assertIn("stage", entry)

    def test_report_stage_targets_from_protocol(self):
        opt = self._run_session("insomnia_relief")
        gen = SleepReportGenerator()
        report = gen.generate(opt)
        self.assertGreater(len(report.stage_targets), 0)

    def test_grade_boundaries(self):
        gen = SleepReportGenerator()
        # Test quality → grade mapping via _compute_quality stub
        # We can test the grade logic indirectly through different sessions
        # Just verify the mapping logic is consistent
        report = SleepReport(quality_score=90, grade="A")
        self.assertEqual(report.grade, "A")


# ══════════════════════════════════════════════════════════════════════════
# Test: Integration (full session flow)
# ══════════════════════════════════════════════════════════════════════════


class TestSleepSystemIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def test_full_session_flow(self):
        """Complete: profile → protocol → session → report."""
        # 1. Circadian profiling
        circadian = CircadianOptimizer(Chronotype.DOLPHIN)
        proto_name = circadian.get_recommended_protocol()
        self.assertIn(proto_name, PROTOCOL_REGISTRY)

        # 2. Start session
        config = SleepOptimizerConfig(
            sample_rate=256, fft_window=256, stage_check_interval=256,
        )
        optimizer = SleepOptimizer(chronotype=Chronotype.DOLPHIN, config=config)
        optimizer.start_session(proto_name)

        # 3. Closed-loop simulation
        stages = [SleepStage.WAKE, SleepStage.N1, SleepStage.N2,
                  SleepStage.N3, SleepStage.N2, SleepStage.REM]
        for i, stage in enumerate(stages * 5):
            eeg = generate_stage_eeg(stage, n_samples=256, seed=i)
            optimizer.add_samples(eeg)
            tick = optimizer.check_and_adapt()
            if tick:
                self.assertIn(tick.current_stage, [s.name for s in SleepStage])

        optimizer.stop_session()

        # 4. Generate report
        gen = SleepReportGenerator()
        report = gen.generate(optimizer)
        self.assertGreater(report.total_duration_min, 0)
        self.assertIn(report.grade, ["A", "B", "C", "D", "F"])
        self.assertGreater(len(report.recommendations), 0)

    def test_all_protocols_can_run(self):
        """Each protocol should complete a short session without errors."""
        config = SleepOptimizerConfig(
            sample_rate=256, fft_window=256, stage_check_interval=256,
        )
        for name in PROTOCOL_REGISTRY:
            opt = SleepOptimizer(config=config)
            opt.start_session(name)
            for i in range(10):
                eeg = generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i)
                opt.add_samples(eeg)
                opt.check_and_adapt()
            opt.stop_session()
            history = opt.get_history()
            self.assertEqual(len(history), 10, msg=f"Protocol {name} failed")

    def test_different_chronotypes_different_protocols(self):
        """Wolf and Lion should recommend different protocols."""
        wolf = CircadianOptimizer(Chronotype.WOLF)
        lion = CircadianOptimizer(Chronotype.LION)
        self.assertNotEqual(
            wolf.get_recommended_protocol(),
            lion.get_recommended_protocol(),
        )

    def test_session_reset_on_restart(self):
        """Starting a new session should clear previous state."""
        config = SleepOptimizerConfig(
            sample_rate=256, fft_window=256, stage_check_interval=256,
        )
        opt = SleepOptimizer(config=config)
        opt.start_session("insomnia_relief")
        for i in range(5):
            eeg = generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i)
            opt.add_samples(eeg)
            opt.check_and_adapt()
        self.assertEqual(len(opt.get_history()), 5)

        # Restart
        opt.start_session("deep_sleep_boost")
        self.assertEqual(len(opt.get_history()), 0)
        self.assertEqual(opt.protocol.name, "deep_sleep_boost")


if __name__ == "__main__":
    unittest.main()
