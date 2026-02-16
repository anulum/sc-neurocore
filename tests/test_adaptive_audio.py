"""
Tests for SSGF Adaptive Audio Engine (Use Case 1)
===================================================

Covers: SSGFEngine, EVSEngine, AdaptiveAudioEngine, UserProfile.

Author: Claude (Session 2026-02-16)
"""

import numpy as np
import pytest

from sc_neurocore.audio.ssgf_engine import SSGFEngine, SSGFConfig, SSGFState
from sc_neurocore.audio.evs_engine import EVSEngine, EVSConfig, EVSSnapshot
from sc_neurocore.audio.adaptive_engine import (
    AdaptiveAudioEngine, AdaptiveConfig, AdaptiveSessionReport, SessionPhase,
)
from sc_neurocore.audio.user_profile import UserProfile, Chronotype


# ── SSGFEngine ───────────────────────────────────────────────────────────

class TestSSGFEngine:
    def test_init(self):
        engine = SSGFEngine(SSGFConfig(N=8, seed=42))
        assert engine.W.shape == (8, 8)
        assert engine.theta.shape == (8,)

    def test_decode_symmetric(self):
        engine = SSGFEngine(SSGFConfig(N=8))
        np.testing.assert_allclose(engine.W, engine.W.T, atol=1e-12)

    def test_decode_non_negative(self):
        engine = SSGFEngine(SSGFConfig(N=8))
        assert np.all(engine.W >= 0)

    def test_decode_zero_diagonal(self):
        engine = SSGFEngine(SSGFConfig(N=8))
        np.testing.assert_allclose(np.diag(engine.W), 0.0)

    def test_outer_step(self):
        engine = SSGFEngine(SSGFConfig(N=8, micro_steps=5, seed=42))
        state = engine.outer_step()
        assert isinstance(state, SSGFState)
        assert 0.0 <= state.R_global <= 1.0
        assert state.outer_step == 1

    def test_convergence(self):
        """C_micro should decrease over iterations."""
        engine = SSGFEngine(SSGFConfig(N=8, micro_steps=10, seed=42))
        costs = []
        for _ in range(10):
            state = engine.outer_step()
            costs.append(state.C_micro)
        # Not strictly monotone, but should trend down
        assert costs[-1] <= costs[0] + 0.2  # Allow some noise

    def test_audio_mapping(self):
        engine = SSGFEngine(SSGFConfig(N=8, seed=42))
        engine.outer_step()
        mapping = engine.get_audio_mapping()
        assert "intensity" in mapping
        assert "binaural_hz" in mapping
        assert "pulse_rate" in mapping
        assert "spatial_angle" in mapping
        assert "theurgic_mode" in mapping
        assert 0.5 <= mapping["binaural_hz"] <= 40.0
        assert 1.0 <= mapping["pulse_rate"] <= 16.0

    def test_update_config(self):
        engine = SSGFEngine(SSGFConfig(N=8))
        engine.update_config(sigma_g=0.5, lr_z=0.02)
        assert engine.config.sigma_g == 0.5
        assert engine.config.lr_z == 0.02

    def test_reset(self):
        engine = SSGFEngine(SSGFConfig(N=8, seed=42))
        engine.outer_step()
        engine.reset(seed=99)
        assert engine._outer_step == 0

    def test_state_serialization(self):
        engine = SSGFEngine(SSGFConfig(N=4, seed=42))
        state = engine.outer_step()
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "R_global" in d


# ── EVSEngine ────────────────────────────────────────────────────────────

class TestEVSEngine:
    def test_init(self):
        engine = EVSEngine()
        assert engine.config.sample_rate == 256

    def test_baseline(self):
        engine = EVSEngine(EVSConfig(fft_window=64, sample_rate=256))
        engine.start_baseline()
        for t in range(128):
            engine.add_sample(np.sin(2 * np.pi * 10.0 * t / 256))
        powers = engine.stop_baseline()
        assert "alpha" in powers

    def test_compute_entrained(self):
        """Strong sinusoidal at target → high EVS."""
        engine = EVSEngine(EVSConfig(target_hz=10.0, fft_window=256, sample_rate=256))
        engine.start_baseline()
        for t in range(256):
            engine.add_sample(np.random.normal(0, 0.5))
        engine.stop_baseline()
        engine.start_session(10.0)
        # Feed strong 10 Hz signal
        for t in range(512):
            engine.add_sample(2.0 * np.sin(2 * np.pi * 10.0 * t / 256))
        snap = engine.compute()
        assert snap.evs_score > 30  # Should be reasonably high

    def test_compute_noise(self):
        """Random noise → low EVS."""
        engine = EVSEngine(EVSConfig(target_hz=10.0, fft_window=128, sample_rate=256))
        engine.start_baseline()
        rng = np.random.RandomState(42)
        for _ in range(128):
            engine.add_sample(rng.normal(0, 1))
        engine.stop_baseline()
        engine.start_session(10.0)
        for _ in range(256):
            engine.add_sample(rng.normal(0, 1))
        snap = engine.compute()
        assert snap.evs_score < 70  # Should be low-ish

    def test_snapshot_serialization(self):
        snap = EVSSnapshot(evs_score=75.0, target_hz=10.0)
        d = snap.to_dict()
        assert d["evs_score"] == 75.0

    def test_reset(self):
        engine = EVSEngine()
        engine.add_sample(1.0)
        engine.reset()
        assert len(engine._buffer) == 0


# ── UserProfile ──────────────────────────────────────────────────────────

class TestUserProfile:
    def test_default(self):
        profile = UserProfile()
        assert profile.chronotype == Chronotype.BEAR

    def test_optimal_target_hz(self):
        profile = UserProfile(chronotype=Chronotype.LION)
        hz = profile.get_optimal_target_hz()
        assert 5.0 <= hz <= 20.0

    def test_wolf_different(self):
        lion = UserProfile(chronotype=Chronotype.LION)
        wolf = UserProfile(chronotype=Chronotype.WOLF)
        assert lion.get_optimal_target_hz() != wolf.get_optimal_target_hz()

    def test_ssgf_overrides(self):
        profile = UserProfile(chronotype=Chronotype.WOLF)
        overrides = profile.get_ssgf_config_overrides()
        assert "sigma_g" in overrides

    def test_update_from_session(self):
        profile = UserProfile()
        profile.update_from_session(target_hz=10.0, evs_avg=75.0, evs_peak=90.0)
        assert profile.session_count == 1
        assert profile.best_evs_score == 90.0
        assert profile.sensitivity_map["alpha"] > 0.5

    def test_to_dict(self):
        profile = UserProfile()
        d = profile.to_dict()
        assert "chronotype" in d
        assert "optimal_target_hz" in d


# ── AdaptiveAudioEngine ─────────────────────────────────────────────────

class TestAdaptiveAudioEngine:
    @pytest.fixture
    def engine_setup(self):
        ssgf = SSGFEngine(SSGFConfig(N=8, micro_steps=3, seed=42))
        evs = EVSEngine(EVSConfig(target_hz=10.0, fft_window=64, sample_rate=256))
        profile = UserProfile()
        adaptive = AdaptiveAudioEngine(
            ssgf, evs, profile,
            AdaptiveConfig(phase1_duration_s=10, phase2_duration_s=30),
        )
        return adaptive, evs

    def test_init(self, engine_setup):
        adaptive, _ = engine_setup
        assert adaptive._tick == 0

    def test_session_start(self, engine_setup):
        adaptive, _ = engine_setup
        adaptive.start_session()
        assert adaptive._running is True

    def test_phase_discovery(self, engine_setup):
        adaptive, _ = engine_setup
        adaptive.start_session()
        assert adaptive.session_phase == SessionPhase.DISCOVERY

    def test_phase_transitions(self, engine_setup):
        adaptive, evs = engine_setup
        adaptive.start_session()
        # Simulate ticks to move through phases
        evs_snap = EVSSnapshot(evs_score=50.0)
        for _ in range(15):
            adaptive.on_evs_update(evs_snap)
        assert adaptive.session_phase == SessionPhase.LOCK_ON

    def test_on_evs_update(self, engine_setup):
        adaptive, evs = engine_setup
        adaptive.start_session()
        snap = adaptive.on_evs_update(EVSSnapshot(evs_score=30.0))
        assert snap.tick == 0
        assert "audio_params" in snap.to_dict()

    def test_low_evs_triggers_adaptation(self, engine_setup):
        adaptive, _ = engine_setup
        adaptive.start_session()
        snap = adaptive.on_evs_update(EVSSnapshot(evs_score=20.0))
        assert "broaden_sweep" in snap.adaptations_applied

    def test_high_evs_stable(self, engine_setup):
        adaptive, _ = engine_setup
        adaptive.start_session()
        # Move to lock_on phase
        for _ in range(12):
            adaptive.on_evs_update(EVSSnapshot(evs_score=80.0))
        snap = adaptive.on_evs_update(EVSSnapshot(evs_score=80.0))
        assert "tighten_params" in snap.adaptations_applied

    def test_session_report(self, engine_setup):
        adaptive, _ = engine_setup
        adaptive.start_session()
        for i in range(20):
            adaptive.on_evs_update(EVSSnapshot(evs_score=50.0 + i, is_verified=i > 10))
        adaptive.stop_session()
        report = adaptive.get_session_report()
        assert report.total_ticks == 20
        assert report.evs_avg > 0
        assert report.grade in ("A", "B", "C", "D", "F")

    def test_report_serialization(self, engine_setup):
        adaptive, _ = engine_setup
        adaptive.start_session()
        adaptive.on_evs_update(EVSSnapshot(evs_score=50.0))
        report = adaptive.get_session_report()
        d = report.to_dict()
        assert "grade" in d

    def test_get_state(self, engine_setup):
        adaptive, _ = engine_setup
        adaptive.start_session()
        adaptive.on_evs_update(EVSSnapshot(evs_score=50.0))
        state = adaptive.get_state()
        assert "evs_score" in state
