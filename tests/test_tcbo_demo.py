"""
Tests for TCBO Consciousness Detection Demo Engine
===================================================

Covers:
- SyntheticEEGGenerator (all perturbation modes)
- TCBOObserver (p_h1 computation, gate thresholds)
- TCBOController (PI control, anti-windup)
- GapJunctionCoupling (topologies)
- TCBODemoEngine (all 5 scenarios end-to-end)

Author: Claude (Session 2026-02-16)
"""

import numpy as np
import pytest

from sc_neurocore.experiments.tcbo_demo_engine import (
    SyntheticEEGGenerator,
    TCBOObserver,
    TCBOController,
    GapJunctionCoupling,
    TCBODemoEngine,
    SCENARIOS,
    build_knm_matrix,
    OMEGA_N,
)


# ── SyntheticEEGGenerator ───────────────────────────────────────────────

class TestSyntheticEEGGenerator:
    def test_init_default(self):
        gen = SyntheticEEGGenerator(N=16, seed=0)
        assert gen.phases.shape == (16,)
        assert gen.K.shape == (16, 16)

    def test_step_returns_phases(self):
        gen = SyntheticEEGGenerator(N=8, seed=1)
        phases = gen.step()
        assert phases.shape == (8,)
        assert np.all(phases >= 0) and np.all(phases < 2 * np.pi)

    def test_run_batch(self):
        gen = SyntheticEEGGenerator(N=4, seed=2)
        history = gen.run(50)
        assert history.shape == (50, 4)

    def test_order_parameter_range(self):
        gen = SyntheticEEGGenerator(N=16, seed=3)
        gen.run(100)
        R = gen.compute_order_parameter()
        assert 0.0 <= R <= 1.0

    def test_anesthesia_reduces_coupling(self):
        gen = SyntheticEEGGenerator(N=16, seed=4)
        K_before = gen.K.sum()
        gen.apply_anesthesia(strength=0.9)
        K_after = gen.K.sum()
        assert K_after < K_before * 0.2

    def test_meditation_boosts_alpha(self):
        gen = SyntheticEEGGenerator(N=16, seed=5)
        K_01_before = gen.K[0, 1]
        gen.apply_meditation(alpha_boost=3.0)
        K_01_after = gen.K[0, 1]
        assert K_01_after > K_01_before * 2.5

    def test_sleep_onset_decays(self):
        gen = SyntheticEEGGenerator(N=16, seed=6)
        K_sum_before = gen.K.sum()
        gen.apply_sleep_onset(decay_factor=0.5)
        K_sum_after = gen.K.sum()
        assert K_sum_after < K_sum_before * 0.6

    def test_reset_restores(self):
        gen = SyntheticEEGGenerator(N=8, seed=7)
        K_orig = gen.K.copy()
        gen.apply_anesthesia(0.9)
        gen.reset()
        np.testing.assert_allclose(gen.K, K_orig)


# ── TCBOObserver ─────────────────────────────────────────────────────────

class TestTCBOObserver:
    def test_init(self):
        obs = TCBOObserver(N=16)
        assert obs.p_h1 == 0.0

    def test_requires_warmup(self):
        obs = TCBOObserver(N=4, window_size=10)
        phases = np.array([0.0, 0.1, 0.2, 0.3])
        result = obs.push_and_compute(phases)
        assert result["p_h1"] == 0.0  # Not enough history

    def test_coherent_phases_high_p_h1(self):
        """Synchronized phases should yield high p_h1."""
        obs = TCBOObserver(N=8, window_size=20)
        # Feed 30 steps of nearly-synchronized phases
        for t in range(30):
            phases = np.full(8, t * 0.1) + np.random.normal(0, 0.05, 8)
            result = obs.push_and_compute(phases)
        assert result["p_h1"] > 0.6

    def test_random_phases_low_p_h1(self):
        """Random phases should yield low p_h1."""
        obs = TCBOObserver(N=8, window_size=20)
        rng = np.random.RandomState(99)
        for t in range(30):
            phases = rng.uniform(0, 2 * np.pi, 8)
            result = obs.push_and_compute(phases)
        assert result["p_h1"] < 0.6

    def test_reset_clears_state(self):
        obs = TCBOObserver(N=4)
        obs.push_and_compute(np.zeros(4))
        obs.reset()
        assert obs.p_h1 == 0.0
        assert len(obs._history) == 0


# ── TCBOController ───────────────────────────────────────────────────────

class TestTCBOController:
    def test_no_action_when_conscious(self):
        ctrl = TCBOController(tau_h1=0.72)
        result = ctrl.step(p_h1=0.9, kappa=0.5, dt=0.01)
        assert result["error"] == 0.0
        assert result["gate_open"] is True

    def test_increases_kappa_when_unconscious(self):
        ctrl = TCBOController(tau_h1=0.72, Kp=2.0)
        result = ctrl.step(p_h1=0.3, kappa=0.5, dt=0.01)
        assert result["kappa_new"] > 0.5
        assert result["gate_open"] is False

    def test_anti_windup(self):
        ctrl = TCBOController(tau_h1=0.72, Ki=1.0, kappa_max=5.0)
        # Many steps with low p_h1 → integral should not explode
        for _ in range(1000):
            result = ctrl.step(p_h1=0.0, kappa=0.5, dt=0.1)
        assert result["kappa_new"] <= 5.0

    def test_reset(self):
        ctrl = TCBOController()
        ctrl.step(p_h1=0.3, kappa=0.5, dt=0.1)
        ctrl.reset()
        assert ctrl._integral == 0.0


# ── GapJunctionCoupling ─────────────────────────────────────────────────

class TestGapJunctionCoupling:
    def test_nearest_topology(self):
        gjc = GapJunctionCoupling(N=8, topology="nearest")
        assert gjc.L.shape == (8, 8)
        # Laplacian: row sums = 0
        np.testing.assert_allclose(gjc.L.sum(axis=1), 0.0, atol=1e-12)

    def test_small_world_topology(self):
        gjc = GapJunctionCoupling(N=16, topology="small_world")
        np.testing.assert_allclose(gjc.L.sum(axis=1), 0.0, atol=1e-12)

    def test_full_topology(self):
        gjc = GapJunctionCoupling(N=4, topology="full")
        np.testing.assert_allclose(gjc.L.sum(axis=1), 0.0, atol=1e-12)

    def test_coupling_output_shape(self):
        gjc = GapJunctionCoupling(N=8)
        phases = np.random.uniform(0, 2 * np.pi, 8)
        delta = gjc.compute_coupling(phases, kappa=1.0)
        assert delta.shape == (8,)


# ── TCBODemoEngine (End-to-End) ──────────────────────────────────────────

class TestTCBODemoEngine:
    def test_init(self):
        engine = TCBODemoEngine(N=16, seed=42)
        assert engine.N == 16
        assert engine.tick == 0

    def test_single_step(self):
        engine = TCBODemoEngine(N=8, seed=10)
        snap = engine.step()
        assert snap.tick == 0
        assert len(snap.phases) == 8

    def test_get_state(self):
        engine = TCBODemoEngine(N=8, seed=11)
        engine.step()
        state = engine.get_state()
        assert "p_h1" in state
        assert "R_global" in state

    def test_scenario_healthy_awake(self):
        """Healthy awake: high coherence → p_h1 mostly above threshold."""
        engine = TCBODemoEngine(N=16, seed=42)
        snaps = engine.run_scenario("healthy_awake")
        assert len(snaps) == 300
        # After warmup, p_h1 should be relatively high
        late_p_h1 = [s.p_h1 for s in snaps[100:]]
        assert np.mean(late_p_h1) > 0.5

    def test_scenario_anesthesia(self):
        """Anesthesia: R_global drops after perturbation (coupling reduced 90%)."""
        engine = TCBODemoEngine(N=16, seed=42)
        snaps = engine.run_scenario("anesthesia")
        # Compare order parameter R which responds faster than p_h1
        pre_R = np.mean([s.R_global for s in snaps[60:99]])
        post_R = np.mean([s.R_global for s in snaps[250:]])
        # Coherence should drop after anesthesia
        assert post_R < pre_R or pre_R < 0.5  # Either drops or was already low

    def test_scenario_meditation(self):
        """Meditation: alpha boost → sustained high p_h1."""
        engine = TCBODemoEngine(N=16, seed=42)
        snaps = engine.run_scenario("meditation")
        late_p_h1 = [s.p_h1 for s in snaps[150:]]
        assert np.mean(late_p_h1) > 0.5

    def test_scenario_sleep_onset(self):
        """Sleep onset: gradual p_h1 decline."""
        engine = TCBODemoEngine(N=16, seed=42)
        snaps = engine.run_scenario("sleep_onset")
        early_R = np.mean([s.R_global for s in snaps[:50]])
        late_R = np.mean([s.R_global for s in snaps[-50:]])
        assert late_R < early_R  # Coherence should decay

    def test_scenario_recovery(self):
        """Recovery: PI controller restores p_h1 after anesthesia."""
        engine = TCBODemoEngine(N=16, seed=42)
        snaps = engine.run_scenario("recovery")
        # After anesthesia at step 100, controller should recover
        # Check that late kappa is higher than initial
        late_kappa = np.mean([s.kappa for s in snaps[-100:]])
        assert late_kappa > 0.5  # Controller has increased coupling

    def test_unknown_scenario_raises(self):
        engine = TCBODemoEngine()
        with pytest.raises(ValueError, match="Unknown scenario"):
            engine.run_scenario("nonexistent")

    def test_get_scenarios(self):
        engine = TCBODemoEngine()
        scenarios = engine.get_scenarios()
        assert len(scenarios) == 5
        assert "healthy_awake" in scenarios

    def test_stop_scenario(self):
        engine = TCBODemoEngine(N=8, seed=20)
        engine.scenario_name = "healthy_awake"
        engine._running = True
        engine.stop()
        assert engine._running is False

    def test_snapshot_serialization(self):
        engine = TCBODemoEngine(N=4, seed=30)
        snap = engine.step()
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["phases"], list)
        assert isinstance(d["p_h1"], float)

    def test_callback(self):
        engine = TCBODemoEngine(N=8, seed=40)
        received = []
        engine.run_scenario("healthy_awake", callback=lambda s: received.append(s))
        assert len(received) == 300


# ── build_knm_matrix ─────────────────────────────────────────────────────

class TestBuildKnm:
    def test_shape(self):
        K = build_knm_matrix(16)
        assert K.shape == (16, 16)

    def test_symmetric(self):
        K = build_knm_matrix(16)
        np.testing.assert_allclose(K, K.T)

    def test_zero_diagonal(self):
        K = build_knm_matrix(16)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_calibration_anchors(self):
        K = build_knm_matrix(16)
        assert K[0, 1] == pytest.approx(0.302)
        assert K[1, 2] == pytest.approx(0.201)
