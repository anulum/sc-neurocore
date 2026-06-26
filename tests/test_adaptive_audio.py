# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SSGF Adaptive Audio (UC1) — 32 tests

"""Tests for SSGF Adaptive Audio (UC1) — 32 tests."""

from __future__ import annotations
import unittest
import numpy as np

from sc_neurocore.audio import (
    SSGFEngine,
    EVSEngine,
    EVSSnapshot,
    AdaptiveAudioEngine,
    AdaptiveSessionReport,
    UserProfile,
    Chronotype,
)
from sc_neurocore.audio.ssgf_engine import SSGFConfig
from sc_neurocore.audio.evs_engine import EVSConfig
from sc_neurocore.audio.adaptive_engine import SessionPhase


def _snapshot(
    *,
    evs_score: float = 60.0,
    is_verified: bool = True,
    peak_alignment: float = 0.7,
    target_hz: float = 10.0,
    peak_hz: float = 10.2,
) -> EVSSnapshot:
    return EVSSnapshot(
        evs_score=evs_score,
        relative_increase=0.5,
        peak_alignment=peak_alignment,
        band_dominance=0.3,
        temporal_consistency=0.8,
        is_verified=is_verified,
        confidence=0.7,
        target_hz=target_hz,
        peak_hz=peak_hz,
        band_powers={"alpha": 0.5},
        timestamp=0,
    )


def _small_adaptive_engine() -> AdaptiveAudioEngine:
    ssgf = SSGFEngine(SSGFConfig(N=4, z_dim=6, micro_steps=1, seed=7))
    return AdaptiveAudioEngine(ssgf, EVSEngine(), UserProfile())


class TestSSGFEngine(unittest.TestCase):
    def test_init(self) -> None:
        eng = SSGFEngine()
        self.assertEqual(eng.theta.shape[0], 16)

    def test_outer_step(self) -> None:
        eng = SSGFEngine()
        cost = eng.outer_step()
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0)

    def test_W_symmetric(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        W = eng.W
        self.assertTrue(np.allclose(W, W.T, atol=1e-10))

    def test_W_non_negative(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        self.assertTrue(np.all(eng.W >= -1e-10))

    def test_W_zero_diagonal(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        self.assertTrue(np.allclose(np.diag(eng.W), 0))

    def test_audio_mapping(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        m = eng.get_audio_mapping()
        self.assertIn("binaural_hz", m)
        self.assertIn("intensity", m)
        self.assertGreaterEqual(m["binaural_hz"], 0.5)
        self.assertLessEqual(m["binaural_hz"], 40.0)

    def test_convergence(self) -> None:
        eng = SSGFEngine(SSGFConfig(seed=42))
        costs = [eng.outer_step() for _ in range(20)]
        self.assertLess(costs[-1], costs[0] * 2)  # Should not diverge

    def test_state(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        s = eng.get_state()
        self.assertIn("R_global", s)
        self.assertIn("audio", s)
        self.assertIn("eigvals", s)

    def test_R_range(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        self.assertGreaterEqual(eng.get_state()["R_global"], 0)
        self.assertLessEqual(eng.get_state()["R_global"], 1)


class TestEVSEngine(unittest.TestCase):
    def test_init(self) -> None:
        eng = EVSEngine()
        self.assertFalse(eng.baseline_done)

    def test_baseline(self) -> None:
        eng = EVSEngine(EVSConfig(sample_rate=256, fft_window=256, baseline_duration_s=0.5))
        eng.start_baseline()
        for v in np.sin(np.linspace(0, 20 * np.pi, 256)):
            eng.add_sample(float(v))
        self.assertTrue(len(eng._baseline_samples) > 0 or eng._baseline_done)

    def test_compute_returns_snapshot(self) -> None:
        cfg = EVSConfig(
            sample_rate=256, fft_window=256, baseline_duration_s=0.5, update_interval_samples=64
        )
        eng = EVSEngine(cfg)
        eng.start_baseline()
        for v in np.sin(np.linspace(0, 20 * np.pi, 256)):
            eng.add_sample(float(v))
        eng.set_target(10.0)
        for v in np.sin(np.linspace(0, 20 * np.pi, 256)):
            eng.add_sample(float(v))
        snap = eng.compute()
        if snap is not None:
            self.assertIsInstance(snap, EVSSnapshot)
            self.assertGreaterEqual(snap.evs_score, 0)
            self.assertLessEqual(snap.evs_score, 100)

    def test_score_range(self) -> None:
        cfg = EVSConfig(
            sample_rate=256, fft_window=256, baseline_duration_s=0.5, update_interval_samples=64
        )
        eng = EVSEngine(cfg)
        eng.start_baseline()
        rng = np.random.RandomState(42)
        for v in rng.randn(256):
            eng.add_sample(float(v))
        eng.set_target(10.0)
        for v in rng.randn(256):
            eng.add_sample(float(v))
        snap = eng.compute()
        if snap:
            self.assertGreaterEqual(snap.evs_score, 0)
            self.assertLessEqual(snap.evs_score, 100)


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
        eng.on_evs_update(_snapshot(evs_score=30.0, peak_alignment=0.4, target_hz=10.0, peak_hz=20.0))

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


class TestUserProfile(unittest.TestCase):
    def test_init(self) -> None:
        p = UserProfile()
        self.assertEqual(p.chronotype, Chronotype.BEAR)

    def test_chronotypes(self) -> None:
        for ct in Chronotype:
            p = UserProfile(chronotype=ct)
            self.assertEqual(p.chronotype, ct)

    def test_get_best_target(self) -> None:
        p = UserProfile()
        hz = p.get_best_target_hz()
        self.assertGreater(hz, 0)

    def test_update_from_session(self) -> None:
        p = UserProfile()
        # update_from_session(avg_evs, peak_evs, best_target_hz=None, band_powers=None)
        p.update_from_session(avg_evs=65.0, peak_evs=80.0)
        self.assertEqual(p.session_count, 1)

    def test_to_dict(self) -> None:
        d = UserProfile().to_dict()
        self.assertIn("chronotype", d)
        self.assertIn("user_id", d)

    def test_from_dict(self) -> None:
        p = UserProfile(chronotype=Chronotype.WOLF, user_id="test")
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        self.assertEqual(p2.chronotype, Chronotype.WOLF)
        self.assertEqual(p2.user_id, "test")

    def test_preferred_cost_weights(self) -> None:
        for ct in Chronotype:
            p = UserProfile(chronotype=ct)
            w = p.preferred_cost_weights
            self.assertIn("w_micro", w)
            self.assertGreater(w["w_micro"], 0)

    def test_avg_session_evs_after_update(self) -> None:
        p = UserProfile()
        p.update_from_session(avg_evs=65.0, peak_evs=80.0)
        self.assertEqual(p.session_count, 1)


class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self) -> None:
        profile = UserProfile(chronotype=Chronotype.BEAR)
        ssgf = SSGFEngine()
        evs = EVSEngine()
        adaptive = AdaptiveAudioEngine(ssgf, evs, profile)

        snap = EVSSnapshot(
            evs_score=55.0,
            relative_increase=0.4,
            peak_alignment=0.6,
            band_dominance=0.25,
            temporal_consistency=0.7,
            is_verified=True,
            confidence=0.65,
            target_hz=10.0,
            peak_hz=10.5,
            band_powers={"alpha": 0.4},
            timestamp=0,
        )
        for _ in range(20):
            result = adaptive.on_evs_update(snap)
            self.assertIn("binaural_hz", result)

        report = adaptive.get_session_report()
        self.assertGreater(report.total_ticks, 0)

    def test_ssgf_audio_consistency(self) -> None:
        eng = SSGFEngine(SSGFConfig(seed=42))
        for _ in range(5):
            eng.outer_step()
        m = eng.get_audio_mapping()
        self.assertGreaterEqual(m["binaural_hz"], 0.5)
        self.assertLessEqual(m["binaural_hz"], 40.0)
        self.assertGreaterEqual(m["intensity"], 0.0)
        self.assertLessEqual(m["intensity"], 1.0)


if __name__ == "__main__":
    unittest.main()
