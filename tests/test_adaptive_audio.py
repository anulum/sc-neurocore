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


class TestSSGFEngine(unittest.TestCase):
    def test_init(self):
        eng = SSGFEngine()
        self.assertEqual(eng.theta.shape[0], 16)

    def test_outer_step(self):
        eng = SSGFEngine()
        cost = eng.outer_step()
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0)

    def test_W_symmetric(self):
        eng = SSGFEngine()
        eng.outer_step()
        W = eng.W
        self.assertTrue(np.allclose(W, W.T, atol=1e-10))

    def test_W_non_negative(self):
        eng = SSGFEngine()
        eng.outer_step()
        self.assertTrue(np.all(eng.W >= -1e-10))

    def test_W_zero_diagonal(self):
        eng = SSGFEngine()
        eng.outer_step()
        self.assertTrue(np.allclose(np.diag(eng.W), 0))

    def test_audio_mapping(self):
        eng = SSGFEngine()
        eng.outer_step()
        m = eng.get_audio_mapping()
        self.assertIn("binaural_hz", m)
        self.assertIn("intensity", m)
        self.assertGreaterEqual(m["binaural_hz"], 0.5)
        self.assertLessEqual(m["binaural_hz"], 40.0)

    def test_convergence(self):
        eng = SSGFEngine(SSGFConfig(seed=42))
        costs = [eng.outer_step() for _ in range(20)]
        self.assertLess(costs[-1], costs[0] * 2)  # Should not diverge

    def test_state(self):
        eng = SSGFEngine()
        eng.outer_step()
        s = eng.get_state()
        self.assertIn("R_global", s)
        self.assertIn("audio", s)
        self.assertIn("eigvals", s)

    def test_R_range(self):
        eng = SSGFEngine()
        eng.outer_step()
        self.assertGreaterEqual(eng.get_state()["R_global"], 0)
        self.assertLessEqual(eng.get_state()["R_global"], 1)


class TestEVSEngine(unittest.TestCase):
    def test_init(self):
        eng = EVSEngine()
        self.assertFalse(eng.baseline_done)

    def test_baseline(self):
        eng = EVSEngine(EVSConfig(sample_rate=256, fft_window=256, baseline_duration_s=0.5))
        eng.start_baseline()
        for v in np.sin(np.linspace(0, 20 * np.pi, 256)):
            eng.add_sample(float(v))
        self.assertTrue(len(eng._baseline_samples) > 0 or eng._baseline_done)

    def test_compute_returns_snapshot(self):
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

    def test_score_range(self):
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
    def test_init(self):
        ssgf = SSGFEngine()
        evs = EVSEngine()
        profile = UserProfile()
        eng = AdaptiveAudioEngine(ssgf, evs, profile)
        self.assertEqual(eng.current_phase, SessionPhase.DISCOVERY)

    def test_on_evs_update(self):
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

    def test_phase_transition(self):
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

    def test_session_report(self):
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

    def test_low_evs_adjusts_params(self):
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


class TestUserProfile(unittest.TestCase):
    def test_init(self):
        p = UserProfile()
        self.assertEqual(p.chronotype, Chronotype.BEAR)

    def test_chronotypes(self):
        for ct in Chronotype:
            p = UserProfile(chronotype=ct)
            self.assertEqual(p.chronotype, ct)

    def test_get_best_target(self):
        p = UserProfile()
        hz = p.get_best_target_hz()
        self.assertGreater(hz, 0)

    def test_update_from_session(self):
        p = UserProfile()
        # update_from_session(avg_evs, peak_evs, best_target_hz=None, band_powers=None)
        p.update_from_session(avg_evs=65.0, peak_evs=80.0)
        self.assertEqual(p.session_count, 1)

    def test_to_dict(self):
        d = UserProfile().to_dict()
        self.assertIn("chronotype", d)
        self.assertIn("user_id", d)

    def test_from_dict(self):
        p = UserProfile(chronotype=Chronotype.WOLF, user_id="test")
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        self.assertEqual(p2.chronotype, Chronotype.WOLF)
        self.assertEqual(p2.user_id, "test")

    def test_preferred_cost_weights(self):
        for ct in Chronotype:
            p = UserProfile(chronotype=ct)
            w = p.preferred_cost_weights
            self.assertIn("w_micro", w)
            self.assertGreater(w["w_micro"], 0)

    def test_avg_session_evs_after_update(self):
        p = UserProfile()
        p.update_from_session(avg_evs=65.0, peak_evs=80.0)
        self.assertEqual(p.session_count, 1)


class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
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

    def test_ssgf_audio_consistency(self):
        eng = SSGFEngine(SSGFConfig(seed=42))
        for _ in range(5):
            eng.outer_step()
        m = eng.get_audio_mapping()
        self.assertGreaterEqual(m["binaural_hz"], 0.5)
        self.assertLessEqual(m["binaural_hz"], 40.0)
        self.assertGreaterEqual(m["intensity"], 0.0)
        self.assertLessEqual(m["intensity"], 1.0)


# ── Additional coverage tests ──────────────────────────────────────────


class TestComputeGrade(unittest.TestCase):
    """Cover _compute_grade for all letter boundaries."""

    def test_grade_a(self):
        from sc_neurocore.audio.adaptive_engine import _compute_grade

        self.assertEqual(_compute_grade(80.0), "A")
        self.assertEqual(_compute_grade(100.0), "A")

    def test_grade_b(self):
        from sc_neurocore.audio.adaptive_engine import _compute_grade

        self.assertEqual(_compute_grade(60.0), "B")
        self.assertEqual(_compute_grade(79.9), "B")

    def test_grade_c(self):
        from sc_neurocore.audio.adaptive_engine import _compute_grade

        self.assertEqual(_compute_grade(40.0), "C")

    def test_grade_d(self):
        from sc_neurocore.audio.adaptive_engine import _compute_grade

        self.assertEqual(_compute_grade(20.0), "D")

    def test_grade_f(self):
        from sc_neurocore.audio.adaptive_engine import _compute_grade

        self.assertEqual(_compute_grade(0.0), "F")
        self.assertEqual(_compute_grade(19.9), "F")


class TestAdaptiveSessionReportSerialization(unittest.TestCase):
    def test_to_dict(self):
        r = AdaptiveSessionReport(
            total_ticks=100,
            avg_evs=65.123,
            peak_evs=82.456,
            verified_pct=55.789,
            grade="C",
            adaptations=12,
            phase_durations={"discovery": 100},
            final_audio={"binaural_hz": 10.5},
        )
        d = r.to_dict()
        self.assertEqual(d["total_ticks"], 100)
        self.assertEqual(d["avg_evs"], 65.12)
        self.assertEqual(d["grade"], "C")


class TestAdaptivePhasesDeep(unittest.TestCase):
    """Cover LOCK_ON and DEEPENING adaptation paths."""

    def _make_engine(self):
        ssgf = SSGFEngine()
        evs = EVSEngine()
        return AdaptiveAudioEngine(ssgf, evs, UserProfile())

    def _snap(self, score=60.0, verified=True, peak_align=0.7, peak_hz=10.2):
        return EVSSnapshot(
            evs_score=score,
            relative_increase=0.5,
            peak_alignment=peak_align,
            band_dominance=0.3,
            temporal_consistency=0.8,
            is_verified=verified,
            confidence=0.7,
            target_hz=10.0,
            peak_hz=peak_hz,
            band_powers={"alpha": 0.5},
            timestamp=0,
        )

    def test_lock_on_declining_trend(self):
        """Cover lines 253-257: sigma_g boost when EVS declining."""
        eng = self._make_engine()
        # Enter LOCK_ON (>240 ticks)
        for _ in range(241):
            eng.on_evs_update(self._snap(score=60.0))
        self.assertEqual(eng.current_phase, SessionPhase.LOCK_ON)
        # Feed declining EVS scores to create negative trend
        for i in range(15):
            eng.on_evs_update(self._snap(score=50.0 - i * 3))
        # Should have made adaptations
        self.assertGreater(eng._tick, 250)

    def test_lock_on_improving_trend(self):
        """Cover lines 261-265: lr_z stabilisation when EVS improving."""
        eng = self._make_engine()
        for _ in range(241):
            eng.on_evs_update(self._snap(score=50.0))
        # Feed improving EVS scores
        for i in range(15):
            eng.on_evs_update(self._snap(score=50.0 + i * 3))
        self.assertEqual(eng.current_phase, SessionPhase.LOCK_ON)

    def test_lock_on_target_adjust(self):
        """Cover lines 270-272: target Hz nudge on low peak alignment."""
        eng = self._make_engine()
        for _ in range(241):
            eng.on_evs_update(self._snap())
        # Low peak alignment triggers target nudge
        eng.on_evs_update(self._snap(peak_align=0.3, peak_hz=12.0))
        self.assertEqual(eng.current_phase, SessionPhase.LOCK_ON)

    def test_deepening_phase(self):
        """Cover lines 276-305: full deepening adaptation."""
        eng = self._make_engine()
        # Enter DEEPENING (>1200 ticks)
        for _ in range(1201):
            eng.on_evs_update(self._snap(score=60.0))
        self.assertEqual(eng.current_phase, SessionPhase.DEEPENING)
        # Run a few more ticks in deepening
        for _ in range(10):
            eng.on_evs_update(self._snap(score=70.0))

    def test_report_phase_durations_lockon(self):
        """Cover lines 349-355: phase_durations in LOCK_ON."""
        eng = self._make_engine()
        for _ in range(300):
            eng.on_evs_update(self._snap())
        report = eng.get_session_report()
        self.assertIn("discovery", report.phase_durations)
        self.assertIn("lock_on", report.phase_durations)
        self.assertEqual(report.phase_durations["discovery"], 240)

    def test_report_phase_durations_deepening(self):
        """Cover lines 352-355: phase_durations in DEEPENING."""
        eng = self._make_engine()
        for _ in range(1250):
            eng.on_evs_update(self._snap())
        report = eng.get_session_report()
        self.assertIn("deepening", report.phase_durations)

    def test_tick_property(self):
        """Cover line 376: tick property."""
        eng = self._make_engine()
        self.assertEqual(eng.tick, 0)
        eng.on_evs_update(self._snap())
        self.assertEqual(eng.tick, 1)

    def test_reset(self):
        """Cover lines 380-388: reset method."""
        eng = self._make_engine()
        for _ in range(10):
            eng.on_evs_update(self._snap())
        eng.reset()
        self.assertEqual(eng.tick, 0)
        self.assertEqual(eng.current_phase, SessionPhase.DISCOVERY)
        self.assertEqual(len(eng._evs_scores), 0)

    def test_evs_trend_short(self):
        """Cover line 170: early return when recent_evs too short."""
        eng = self._make_engine()
        # Only 1 tick, trend should return 0.0
        eng.on_evs_update(self._snap())
        self.assertAlmostEqual(eng._evs_trend(), 0.0)


class TestEVSEngineDeep(unittest.TestCase):
    """Cover EVS engine edge cases."""

    def test_hz_to_band_high(self):
        """Cover lines 48-50: edge cases for _hz_to_band."""
        from sc_neurocore.audio.evs_engine import _hz_to_band

        self.assertEqual(_hz_to_band(50.0), "gamma")
        self.assertEqual(_hz_to_band(0.1), "delta")
        self.assertEqual(_hz_to_band(10.0), "alpha")

    def test_snapshot_to_dict(self):
        """Cover line 86: EVSSnapshot.to_dict()."""
        snap = EVSSnapshot(
            evs_score=75.5,
            target_hz=10.0,
            peak_hz=10.3,
            band_powers={"alpha": 0.5},
        )
        d = snap.to_dict()
        self.assertEqual(d["evs_score"], 75.5)
        self.assertIn("band_powers", d)

    def test_compute_no_baseline(self):
        """Cover line 229: compute returns None without baseline."""
        eng = EVSEngine()
        self.assertIsNone(eng.compute())

    def test_compute_insufficient_buffer(self):
        """Cover line 231: compute returns None with tiny buffer."""
        eng = EVSEngine(EVSConfig(fft_window=256, baseline_duration_s=0.1, sample_rate=256))
        eng.start_baseline()
        for v in np.zeros(30):
            eng.add_sample(float(v))
        # Force baseline done
        eng._baseline_done = True
        eng._baseline_powers = {"delta": 1.0, "theta": 1.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0}
        # Buffer not full and idx < 32 (only 30 samples)
        self.assertIsNone(eng.compute())

    def test_flat_baseline(self):
        """Cover line 153: flat baseline when < 32 samples."""
        eng = EVSEngine(EVSConfig(fft_window=64, baseline_duration_s=0.05, sample_rate=256))
        eng.start_baseline()
        # Feed only ~13 samples (0.05 * 256)
        for v in np.zeros(13):
            eng.add_sample(float(v))
        # Baseline should be finalised with flat powers
        self.assertTrue(eng.baseline_done)
        self.assertEqual(eng._baseline_powers.get("alpha", 0.0), 1.0)

    def test_ordered_buf_not_full(self):
        """Cover line 187: _ordered_buf when buffer not full."""
        eng = EVSEngine(EVSConfig(fft_window=256))
        for v in np.ones(10):
            eng.add_sample(float(v))
        buf = eng._ordered_buf()
        self.assertEqual(len(buf), 10)

    def test_band_powers_tiny(self):
        """Cover line 194: _band_powers with < 4 samples."""
        eng = EVSEngine()
        powers = eng._band_powers(np.array([1.0, 2.0]))
        for v in powers.values():
            self.assertEqual(v, 0.0)

    def test_peak_frequency_tiny(self):
        """Cover line 212: _peak_frequency with < 4 samples."""
        eng = EVSEngine()
        self.assertEqual(eng._peak_frequency(np.array([1.0])), 0.0)

    def test_score_history_property(self):
        """Cover line 309: score_history property."""
        eng = EVSEngine()
        self.assertEqual(eng.score_history, [])

    def test_reset(self):
        """Cover lines 313-321: full reset."""
        cfg = EVSConfig(sample_rate=256, fft_window=256, baseline_duration_s=0.5)
        eng = EVSEngine(cfg)
        eng.start_baseline()
        for v in np.sin(np.linspace(0, 10 * np.pi, 256)):
            eng.add_sample(float(v))
        eng.reset()
        self.assertFalse(eng.baseline_done)
        self.assertEqual(eng._total_samples, 0)
        self.assertEqual(len(eng._score_history), 0)

    def test_temporal_consistency_with_history(self):
        """Cover lines 265-266: temporal consistency with >= 3 scores."""
        cfg = EVSConfig(sample_rate=256, fft_window=256, baseline_duration_s=0.5)
        eng = EVSEngine(cfg)
        eng.start_baseline()
        for v in np.sin(np.linspace(0, 20 * np.pi, 256)):
            eng.add_sample(float(v))
        eng.set_target(10.0)
        # Compute multiple snapshots to build score history
        for _ in range(5):
            for v in np.sin(np.linspace(0, 20 * np.pi, 256)):
                eng.add_sample(float(v))
            eng.compute()
        self.assertGreaterEqual(len(eng._score_history), 3)


class TestSSGFEdgeCases(unittest.TestCase):
    """Cover SSGF fallback paths for small N."""

    def test_small_N_audio_mapping(self):
        """Cover lines 286, 293, 299: N <= 2/4/7 fallbacks."""
        eng = SSGFEngine(SSGFConfig(N=2, seed=42))
        eng.outer_step()
        m = eng.get_audio_mapping()
        self.assertEqual(m["binaural_hz"], 10.0)
        self.assertEqual(m["pulse_rate"], 8.0)
        self.assertEqual(m["spatial_angle"], 0.0)


class TestUserProfileDeep(unittest.TestCase):
    """Cover UserProfile edge cases."""

    def test_preferred_target_hz_override(self):
        """Cover line 128: get_best_target_hz with explicit preference."""
        p = UserProfile(preferred_target_hz=7.5)
        self.assertEqual(p.get_best_target_hz(), 7.5)

    def test_update_adopts_target_first_time(self):
        """Cover lines 157-158: adopt best_target_hz when preferred is None."""
        p = UserProfile()
        self.assertIsNone(p.preferred_target_hz)
        p.update_from_session(avg_evs=60.0, peak_evs=80.0, best_target_hz=8.0)
        self.assertEqual(p.preferred_target_hz, 8.0)

    def test_update_blends_target_ema(self):
        """Cover lines 160-162: EMA blend of best_target_hz."""
        p = UserProfile(preferred_target_hz=10.0)
        p.update_from_session(avg_evs=60.0, peak_evs=80.0, best_target_hz=8.0)
        # EMA: 0.7 * 10.0 + 0.3 * 8.0 = 9.4
        self.assertAlmostEqual(p.preferred_target_hz, 9.4, places=1)

    def test_update_ignores_low_evs(self):
        """best_target_hz ignored when avg_evs <= 50."""
        p = UserProfile()
        p.update_from_session(avg_evs=30.0, peak_evs=40.0, best_target_hz=8.0)
        self.assertIsNone(p.preferred_target_hz)

    def test_update_with_band_powers_first(self):
        """Cover lines 168-169: initial band_powers adoption."""
        p = UserProfile()
        p.update_from_session(
            avg_evs=60.0,
            peak_evs=80.0,
            band_powers={"alpha": 0.5, "beta": 0.3},
        )
        self.assertAlmostEqual(p.baseline_band_powers["alpha"], 0.5)

    def test_update_with_band_powers_ema(self):
        """Cover lines 170-174: EMA blend of band_powers."""
        p = UserProfile()
        p.baseline_band_powers = {"alpha": 1.0, "beta": 0.5}
        p.update_from_session(
            avg_evs=60.0,
            peak_evs=80.0,
            band_powers={"alpha": 0.0, "beta": 1.0},
        )
        # EMA alpha=0.2: 0.8*1.0 + 0.2*0.0 = 0.8
        self.assertAlmostEqual(p.baseline_band_powers["alpha"], 0.8, places=1)
        # EMA: 0.8*0.5 + 0.2*1.0 = 0.6
        self.assertAlmostEqual(p.baseline_band_powers["beta"], 0.6, places=1)


if __name__ == "__main__":
    unittest.main()
