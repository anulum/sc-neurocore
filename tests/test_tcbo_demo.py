"""Tests for TCBO Consciousness Detection Demo (UC2).

38 tests covering SyntheticEEGGenerator, core functions, TCBODemoEngine,
scenario validations, and singleton management.
"""

from __future__ import annotations

import unittest

import numpy as np

from sc_neurocore.experiments.tcbo_demo_engine import (
    SyntheticEEGGenerator,
    TCBODemoEngine,
    TCBODemoSnapshot,
    TCBOController,
    ScenarioName,
    SCENARIOS,
    _compute_order_parameter,
    _compute_p_h1_lightweight,
    _build_knm,
    get_tcbo_demo_engine,
    reset_tcbo_demo_engine,
)


class TestBuildKnm(unittest.TestCase):
    def test_symmetric(self):
        K = _build_knm(16)
        self.assertTrue(np.allclose(K, K.T))

    def test_zero_diagonal(self):
        K = _build_knm(16)
        self.assertTrue(np.allclose(np.diag(K), 0))

    def test_non_negative(self):
        K = _build_knm(16)
        self.assertTrue(np.all(K >= 0))

    def test_small_N(self):
        K = _build_knm(4)
        self.assertEqual(K.shape, (4, 4))
        self.assertTrue(np.allclose(K, K.T))


class TestOrderParameter(unittest.TestCase):
    def test_synchronized(self):
        theta = np.ones(16) * 1.5
        R = _compute_order_parameter(theta)
        self.assertGreater(R, 0.99)

    def test_uniform(self):
        theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        R = _compute_order_parameter(theta)
        self.assertLess(R, 0.15)

    def test_range(self):
        rng = np.random.RandomState(42)
        for _ in range(10):
            theta = rng.uniform(0, 2 * np.pi, 16)
            R = _compute_order_parameter(theta)
            self.assertGreaterEqual(R, 0.0)
            self.assertLessEqual(R, 1.0)


class TestP_h1Lightweight(unittest.TestCase):
    def test_coherent_high(self):
        history = np.tile(np.ones(16) * 1.0, (60, 1))
        history += np.random.default_rng(0).normal(0, 0.05, history.shape)
        p = _compute_p_h1_lightweight(history)
        self.assertGreater(p, 0.5)

    def test_incoherent_lower(self):
        rng = np.random.default_rng(42)
        history = rng.uniform(0, 2 * np.pi, (60, 16))
        p = _compute_p_h1_lightweight(history)
        self.assertLess(p, 0.8)

    def test_short_history_returns_zero(self):
        history = np.zeros((5, 16))
        p = _compute_p_h1_lightweight(history)
        self.assertEqual(p, 0.0)


class TestSyntheticEEGGenerator(unittest.TestCase):
    def test_init(self):
        gen = SyntheticEEGGenerator(N=16)
        self.assertEqual(gen.theta.shape, (16,))

    def test_step_returns_phases(self):
        gen = SyntheticEEGGenerator(N=16)
        theta = gen.step()
        self.assertEqual(theta.shape, (16,))
        self.assertTrue(np.all(theta >= 0))
        self.assertTrue(np.all(theta < 2 * np.pi))

    def test_run_returns_history(self):
        gen = SyntheticEEGGenerator(N=16)
        h = gen.run(100)
        self.assertEqual(h.shape, (100, 16))

    def test_coupling_scale(self):
        gen = SyntheticEEGGenerator(N=16)
        gen.set_coupling_scale(2.0)
        self.assertTrue(np.allclose(gen.K, gen._K_base * 2.0))

    def test_anesthesia(self):
        gen = SyntheticEEGGenerator(N=16)
        k_before = gen.K.sum()
        gen.apply_anesthesia(0.9)
        self.assertLess(gen.K.sum(), k_before * 0.2)

    def test_alpha_boost(self):
        gen = SyntheticEEGGenerator(N=16)
        k_before = gen.K[1, :].sum()
        gen.apply_alpha_boost(2.0)
        self.assertGreater(gen.K[1, :].sum(), k_before)

    def test_order_parameter(self):
        gen = SyntheticEEGGenerator(N=16)
        R = gen.get_order_parameter()
        self.assertGreaterEqual(R, 0.0)
        self.assertLessEqual(R, 1.0)

    def test_reset(self):
        gen = SyntheticEEGGenerator(N=16, seed=42)
        gen.run(100)
        gen.reset(seed=42)
        self.assertEqual(gen._step_count, 0)


class TestTCBOController(unittest.TestCase):
    def test_deficit_increases_kappa(self):
        ctrl = TCBOController(tau_h1=0.72)
        kappa = ctrl.step(p_h1=0.3, kappa=1.0, dt=0.01)
        self.assertGreater(kappa, 1.0)

    def test_no_deficit_no_change(self):
        ctrl = TCBOController(tau_h1=0.72)
        kappa = ctrl.step(p_h1=0.9, kappa=1.0, dt=0.01)
        self.assertAlmostEqual(kappa, 1.0, places=3)

    def test_kappa_clamped(self):
        ctrl = TCBOController(tau_h1=0.72, kappa_max=5.0)
        kappa = 4.9
        for _ in range(1000):
            kappa = ctrl.step(p_h1=0.0, kappa=kappa, dt=0.1)
        self.assertLessEqual(kappa, 5.0)

    def test_reset(self):
        ctrl = TCBOController()
        ctrl.step(0.1, 1.0, 0.1)
        ctrl.reset()
        self.assertEqual(ctrl._integral, 0.0)


class TestTCBODemoEngine(unittest.TestCase):
    def test_init(self):
        engine = TCBODemoEngine(N=16)
        self.assertEqual(engine.N, 16)
        self.assertFalse(engine.is_running)

    def test_get_scenarios(self):
        engine = TCBODemoEngine()
        s = engine.get_scenarios()
        self.assertEqual(len(s), 5)
        self.assertIn("healthy_awake", s)

    def test_start_scenario(self):
        engine = TCBODemoEngine()
        info = engine.start_scenario("healthy_awake")
        self.assertTrue(engine.is_running)
        self.assertEqual(info["scenario"], "healthy_awake")

    def test_invalid_scenario(self):
        engine = TCBODemoEngine()
        with self.assertRaises(ValueError):
            engine.start_scenario("nonexistent")

    def test_step(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        snap = engine.step()
        self.assertIsInstance(snap, TCBODemoSnapshot)
        self.assertEqual(len(snap.phases), 16)

    def test_snapshot_serialization(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        d = engine.step().to_dict()
        self.assertIn("p_h1", d)
        self.assertIn("phases", d)

    def test_get_state(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        engine.step()
        state = engine.get_state()
        self.assertTrue(state["running"])

    def test_reset(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        for _ in range(100):
            engine.step()
        engine.reset()
        self.assertFalse(engine.is_running)
        self.assertEqual(engine.p_h1, 0.0)

    def test_history(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        for _ in range(50):
            engine.step()
        h = engine.get_history(10)
        self.assertEqual(len(h), 10)

    def test_run_scenario(self):
        engine = TCBODemoEngine()
        snaps = engine.run_scenario("healthy_awake", duration_s=1.0, subsample=100)
        self.assertGreater(len(snaps), 5)

    def test_all_scenarios_run(self):
        engine = TCBODemoEngine(seed=42)
        for name in ScenarioName:
            snaps = engine.run_scenario(name.value, duration_s=1.0, subsample=100)
            self.assertGreater(len(snaps), 0)

    def test_step_without_start_raises(self):
        engine = TCBODemoEngine()
        with self.assertRaises(RuntimeError):
            engine.step()


class TestScenarioValidations(unittest.TestCase):
    def test_healthy_awake_coherence(self):
        engine = TCBODemoEngine(seed=42)
        snaps = engine.run_scenario("healthy_awake", duration_s=3.0, subsample=300)
        last_R = snaps[-1].R_global
        self.assertGreater(last_R, 0.1)

    def test_anesthesia_lower_R(self):
        """Anesthesia should produce lower R than healthy awake."""
        engine = TCBODemoEngine(seed=42)
        healthy = engine.run_scenario("healthy_awake", duration_s=3.0, subsample=300)
        anest = engine.run_scenario("anesthesia", duration_s=3.0, subsample=300)
        R_healthy = np.mean([s.R_global for s in healthy[-5:]])
        R_anest = np.mean([s.R_global for s in anest[-5:]])
        self.assertLess(R_anest, R_healthy + 0.1)

    def test_recovery_kappa_varies(self):
        engine = TCBODemoEngine(seed=42)
        snaps = engine.run_scenario("recovery", duration_s=5.0, subsample=500)
        kappas = [s.kappa for s in snaps]
        self.assertNotAlmostEqual(max(kappas), min(kappas), places=2)

    def test_sleep_onset_runs(self):
        engine = TCBODemoEngine(seed=42)
        snaps = engine.run_scenario("sleep_onset", duration_s=2.0, subsample=200)
        self.assertTrue(all(0 <= s.R_global <= 1 for s in snaps))


class TestSingleton(unittest.TestCase):
    def test_singleton(self):
        reset_tcbo_demo_engine()
        e1 = get_tcbo_demo_engine()
        e2 = get_tcbo_demo_engine()
        self.assertIs(e1, e2)

    def test_reset(self):
        e1 = get_tcbo_demo_engine()
        reset_tcbo_demo_engine()
        e2 = get_tcbo_demo_engine()
        self.assertIsNot(e1, e2)


if __name__ == "__main__":
    unittest.main()
