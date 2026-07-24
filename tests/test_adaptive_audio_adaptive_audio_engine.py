# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveAudioEngine from former test_adaptive_audio.py

"""Focused suite: TestAdaptiveAudioEngine from former test_adaptive_audio.py."""

from __future__ import annotations

from tests.adaptive_audio_support import *  # noqa: F403


class TestAdaptiveAudioEngine(unittest.TestCase):
    def test_init(self) -> None:
        ssgf = SSGFEngine()
        evs = EVSEngine()
        profile = UserProfile()
        eng = AdaptiveAudioEngine(ssgf, evs, profile)
        self.assertEqual(eng.current_phase, SessionPhase.DISCOVERY)

    def test_on_evs_update(self) -> None:
        ssgf = SSGFEngine()
        evs = EVSEngine()
        profile = UserProfile()
        eng = AdaptiveAudioEngine(ssgf, evs, profile)
        snap = EVSSnapshot(
            evs_score=60.0,
            relative_increase=0.5,
            peak_alignment=0.7,
            band_dominance=0.3,
            temporal_consistency=0.8,
            is_verified=True,
            confidence=0.7,
            target_hz=10.0,
            peak_hz=10.2,
            band_powers={"alpha": 0.5},
            timestamp=0,
        )
        result = eng.on_evs_update(snap)
        self.assertIsInstance(result, dict)
        self.assertIn("binaural_hz", result)

    def test_phase_transition(self) -> None:
        ssgf = SSGFEngine()
        evs = EVSEngine()
        profile = UserProfile()
        eng = AdaptiveAudioEngine(ssgf, evs, profile)
        snap = EVSSnapshot(
            evs_score=65.0,
            relative_increase=0.5,
            peak_alignment=0.7,
            band_dominance=0.3,
            temporal_consistency=0.8,
            is_verified=True,
            confidence=0.7,
            target_hz=10.0,
            peak_hz=10.2,
            band_powers={"alpha": 0.5},
            timestamp=0,
        )
        for i in range(250):
            eng.on_evs_update(snap)
        # After 250 ticks (>240 DISCOVERY_TICKS), should be in LOCK_ON
        self.assertNotEqual(eng.current_phase, SessionPhase.DISCOVERY)

    def test_session_report(self) -> None:
        ssgf = SSGFEngine()
        evs = EVSEngine()
        profile = UserProfile()
        eng = AdaptiveAudioEngine(ssgf, evs, profile)
        snap = EVSSnapshot(
            evs_score=70.0,
            relative_increase=0.5,
            peak_alignment=0.7,
            band_dominance=0.3,
            temporal_consistency=0.8,
            is_verified=True,
            confidence=0.7,
            target_hz=10.0,
            peak_hz=10.2,
            band_powers={"alpha": 0.5},
            timestamp=0,
        )
        for _ in range(10):
            eng.on_evs_update(snap)
        report = eng.get_session_report()
        self.assertIsInstance(report, AdaptiveSessionReport)
        self.assertIn(report.grade, "ABCDF")

    def test_low_evs_adjusts_params(self) -> None:
        ssgf = SSGFEngine()
        evs = EVSEngine()
        profile = UserProfile()
        eng = AdaptiveAudioEngine(ssgf, evs, profile)
        low_snap = EVSSnapshot(
            evs_score=20.0,
            relative_increase=0.1,
            peak_alignment=0.2,
            band_dominance=0.1,
            temporal_consistency=0.5,
            is_verified=False,
            confidence=0.3,
            target_hz=10.0,
            peak_hz=15.0,
            band_powers={"alpha": 0.1},
            timestamp=0,
        )
        result = eng.on_evs_update(low_snap)
        self.assertIsInstance(result, dict)

    def test_report_serialization_covers_grade_thresholds(self) -> None:
        report = AdaptiveSessionReport(
            total_ticks=3,
            avg_evs=44.444,
            peak_evs=88.888,
            verified_pct=66.666,
            grade="B",
            adaptations=2,
            phase_durations={"discovery": 3},
            final_audio={"binaural_hz": 10.0},
        )

        self.assertEqual(
            report.to_dict(),
            {
                "total_ticks": 3,
                "avg_evs": 44.44,
                "peak_evs": 88.89,
                "verified_pct": 66.67,
                "grade": "B",
                "adaptations": 2,
                "phase_durations": {"discovery": 3},
                "final_audio": {"binaural_hz": 10.0},
            },
        )

        for verified_pct, expected_grade in [
            (100.0, "A"),
            (75.0, "B"),
            (55.0, "C"),
            (35.0, "D"),
            (10.0, "F"),
        ]:
            eng = _small_adaptive_engine()
            for tick in range(20):
                verified = tick < int(verified_pct / 5.0)
                eng.on_evs_update(_snapshot(evs_score=verified_pct, is_verified=verified))
            self.assertEqual(eng.get_session_report().grade, expected_grade)

    def test_lock_on_declining_trend_boosts_geometry_feedback(self) -> None:
        eng = _small_adaptive_engine()
        for score in [80.0, 70.0, 60.0, 50.0, 40.0]:
            eng.on_evs_update(_snapshot(evs_score=score))

        eng._phase = SessionPhase.LOCK_ON
        before_sigma = eng.ssgf.cfg.sigma_g
        eng.on_evs_update(
            _snapshot(evs_score=30.0, peak_alignment=0.4, target_hz=10.0, peak_hz=20.0)
        )

        self.assertGreater(eng.ssgf.cfg.sigma_g, before_sigma)
        self.assertAlmostEqual(eng.evs._target_hz, 11.0)

    def test_discovery_and_lock_on_respect_parameter_clamps(self) -> None:
        eng = _small_adaptive_engine()
        eng.ssgf.cfg.sigma_g = 0.05
        eng.on_evs_update(_snapshot(evs_score=50.0))
        self.assertAlmostEqual(eng.ssgf.cfg.sigma_g, 0.15)

        for score in [80.0, 70.0, 60.0, 50.0, 40.0]:
            eng.on_evs_update(_snapshot(evs_score=score))
        eng._phase = SessionPhase.LOCK_ON
        eng.ssgf.cfg.sigma_g = 0.6
        eng.on_evs_update(_snapshot(evs_score=30.0))
        self.assertAlmostEqual(eng.ssgf.cfg.sigma_g, 0.6)

    def test_lock_on_improving_trend_stabilises_learning_rate(self) -> None:
        eng = _small_adaptive_engine()
        eng.ssgf.cfg.lr_z = 0.02
        for score in [10.0, 20.0, 30.0, 40.0, 50.0]:
            eng.on_evs_update(_snapshot(evs_score=score))

        eng._phase = SessionPhase.LOCK_ON
        before_lr = eng.ssgf.cfg.lr_z
        eng.on_evs_update(_snapshot(evs_score=60.0))

        self.assertLess(eng.ssgf.cfg.lr_z, before_lr)

    def test_lock_on_learning_rate_floor_is_stable(self) -> None:
        eng = _small_adaptive_engine()
        for score in [10.0, 20.0, 30.0, 40.0, 50.0]:
            eng.on_evs_update(_snapshot(evs_score=score))

        eng._phase = SessionPhase.LOCK_ON
        eng.ssgf.cfg.lr_z = 0.002
        eng.on_evs_update(_snapshot(evs_score=60.0))

        self.assertAlmostEqual(eng.ssgf.cfg.lr_z, 0.002)

    def test_deepening_phase_updates_pressure_and_report_duration(self) -> None:
        eng = _small_adaptive_engine()
        eng._phase = SessionPhase.LOCK_ON
        eng._tick = 1200
        eng.ssgf.R_global = 0.95
        before_pressure = eng.ssgf.cfg.field_pressure
        before_sigma = eng.ssgf.cfg.sigma_g

        result = eng.on_evs_update(_snapshot(evs_score=90.0))

        self.assertEqual(eng.current_phase, SessionPhase.DEEPENING)
        self.assertIn("binaural_hz", result)
        self.assertGreater(eng.ssgf.cfg.field_pressure, before_pressure)
        self.assertGreater(eng.ssgf.cfg.sigma_g, before_sigma)
        report = eng.get_session_report()
        self.assertEqual(report.phase_durations["discovery"], 240)
        self.assertEqual(report.phase_durations["lock_on"], 960)
        self.assertEqual(report.phase_durations["deepening"], 1)

    def test_deepening_phase_respects_clamps(self) -> None:
        eng = _small_adaptive_engine()
        eng._phase = SessionPhase.DEEPENING
        eng.ssgf.cfg.field_pressure = 0.4
        eng.ssgf.cfg.sigma_g = 0.8
        eng.ssgf.cfg.lr_z = 0.001
        eng.ssgf.R_global = 0.5

        eng.on_evs_update(_snapshot(evs_score=90.0))

        self.assertAlmostEqual(eng.ssgf.cfg.field_pressure, 0.4)
        self.assertAlmostEqual(eng.ssgf.cfg.sigma_g, 0.8)
        self.assertAlmostEqual(eng.ssgf.cfg.lr_z, 0.001)

    def test_deepening_near_theurgic_pressure_cap_is_stable(self) -> None:
        eng = _small_adaptive_engine()
        eng._phase = SessionPhase.DEEPENING
        eng.ssgf.cfg.field_pressure = 0.5
        eng.ssgf.R_global = 0.95

        eng.on_evs_update(_snapshot(evs_score=90.0))

        self.assertAlmostEqual(eng.ssgf.cfg.field_pressure, 0.5)

    def test_report_lock_on_duration_and_reset_clear_session_state(self) -> None:
        eng = _small_adaptive_engine()
        for _ in range(3):
            eng.on_evs_update(_snapshot(evs_score=75.0))

        self.assertEqual(eng.tick, 3)
        eng._tick = 300
        report = eng.get_session_report()
        self.assertEqual(report.phase_durations, {"discovery": 240, "lock_on": 60})

        eng.reset()
        self.assertEqual(eng.tick, 0)
        self.assertEqual(eng.current_phase, SessionPhase.DISCOVERY)
        self.assertEqual(eng.get_session_report().total_ticks, 0)
