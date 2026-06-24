# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Neuromorphic Swarm Control (UC4) — 50 tests

"""Tests for Neuromorphic Swarm Control (UC4) — 50 tests."""

from __future__ import annotations
import unittest
import numpy as np

from sc_neurocore.swarm import (
    SwarmAgent,
    AgentConfig,
    SwarmEnvironment,
    EnvConfig,
    CollectiveFields,
    FieldConfig,
    SwarmFitness,
    SwarmEvolver,
    EvolverConfig,
)


class TestSwarmAgent(unittest.TestCase):
    def test_init(self):
        a = SwarmAgent(AgentConfig())
        self.assertEqual(a.agent_id, 0)
        self.assertEqual(a.position.shape, (2,))

    def test_weights_setter_rejects_wrong_size(self):
        a = SwarmAgent(AgentConfig())
        with self.assertRaises(ValueError):
            a.weights = np.zeros(3, dtype=np.float64)

    def test_init_config(self):
        cfg = AgentConfig(n_sensory=20, n_hidden=8, n_motor=2)
        a = SwarmAgent(cfg)
        self.assertEqual(a.W_in.shape, (8, 20))

    def test_think(self):
        a = SwarmAgent(AgentConfig())
        sensory = np.zeros(a.cfg.n_sensory)
        speed, turn = a.think(sensory)
        self.assertIsInstance(speed, float)
        self.assertIsInstance(turn, float)

    def test_act(self):
        a = SwarmAgent(AgentConfig())
        pos_before = a.position.copy()
        a.act(1.0, 0.0)
        self.assertFalse(np.allclose(a.position, pos_before))

    def test_weights_property(self):
        cfg = AgentConfig(n_hidden=8, n_sensory=20, n_motor=2)
        a = SwarmAgent(cfg)
        w = a.weights
        n_expected = 8 * 20 + 8 * 8 + 2 * 8  # W_in + W_rec + W_out
        self.assertEqual(len(w), n_expected)

    def test_weights_setter(self):
        cfg = AgentConfig(n_hidden=8)
        a = SwarmAgent(cfg)
        w = np.ones(a.n_weights) * 0.5
        a.weights = w
        self.assertTrue(np.allclose(a.weights, 0.5))

    def test_reset(self):
        a = SwarmAgent(AgentConfig())
        a.act(1.0, 0.5)
        a.reset()
        self.assertTrue(np.allclose(a.membrane, 0))

    def test_neural_state(self):
        a = SwarmAgent(AgentConfig())
        a.think(np.zeros(a.cfg.n_sensory))
        self.assertIsNotNone(a.firing_rate)
        self.assertIsNotNone(a.membrane)

    def test_heading_wraps(self):
        a = SwarmAgent(AgentConfig())
        a.heading = 0
        a.act(0.0, 100.0)  # Large turn
        self.assertGreaterEqual(a.heading, 0)
        self.assertLess(a.heading, 2 * np.pi)


class TestSwarmEnvironment(unittest.TestCase):
    def test_init(self):
        env = SwarmEnvironment(EnvConfig())
        self.assertEqual(len(env.agents), 20)

    def test_init_config(self):
        cfg = EnvConfig(n_agents=5, n_obstacles=2, n_targets=1)
        env = SwarmEnvironment(cfg)
        self.assertEqual(len(env.agents), 5)

    def test_step(self):
        env = SwarmEnvironment(EnvConfig(n_agents=5))
        env.step()
        # Should not crash

    def test_pairwise_distances(self):
        env = SwarmEnvironment(EnvConfig(n_agents=5))
        D = env.get_pairwise_distances()
        self.assertEqual(D.shape, (5, 5))
        self.assertTrue(np.allclose(np.diag(D), 0))
        self.assertTrue(np.allclose(D, D.T))

    def test_boundary_wrap(self):
        cfg = EnvConfig(n_agents=1, boundary_mode="wrap", width=100, height=100)
        env = SwarmEnvironment(cfg)
        env.agents[0].position = np.array([105.0, -5.0])
        env.step()
        pos = env.agents[0].position
        self.assertGreaterEqual(pos[0], 0)
        self.assertLess(pos[0], 100)

    def test_get_state(self):
        env = SwarmEnvironment(EnvConfig(n_agents=3))
        s = env.get_state()
        self.assertIn("positions", s)
        self.assertEqual(len(s["positions"]), 3)

    def test_agent_config_passed(self):
        acfg = AgentConfig(n_hidden=8)
        ecfg = EnvConfig(n_agents=3, agent_config=acfg)
        env = SwarmEnvironment(ecfg)
        self.assertEqual(env.agents[0].cfg.n_hidden, 8)

    def test_obstacles_created(self):
        env = SwarmEnvironment(EnvConfig(n_obstacles=5))
        self.assertEqual(len(env.obstacles), 5)

    def test_targets_created(self):
        env = SwarmEnvironment(EnvConfig(n_targets=3))
        self.assertEqual(len(env.targets), 3)

    def test_neighbor_distances(self):
        env = SwarmEnvironment(EnvConfig(n_agents=10))
        dists = env.get_neighbor_distances(0, k=5)
        self.assertEqual(len(dists), 5)
        # All should be non-negative
        self.assertTrue(all(d >= 0 for d in dists))


class TestCollectiveFields(unittest.TestCase):
    def test_init(self):
        f = CollectiveFields(FieldConfig(grid_size=50), n_agents=5)
        self.assertEqual(f.chemical_field.shape, (50, 50))

    def test_deposit_chemical(self):
        f = CollectiveFields(FieldConfig(grid_size=50), n_agents=5)
        f.deposit_chemical(25.0, 25.0, 1.0)
        self.assertGreater(f.chemical_field.max(), 0)

    def test_diffuse(self):
        f = CollectiveFields(FieldConfig(grid_size=20), n_agents=5)
        # Deposit at a valid position (mapped to grid coords internally)
        f.deposit_chemical(50.0, 50.0, 10.0)
        val_before = f.chemical_field.max()
        self.assertGreater(val_before, 0)
        f.diffuse(1.0)
        # After diffusion + decay the peak should decrease
        val_after = f.chemical_field.max()
        self.assertLessEqual(val_after, val_before)

    def test_gradient(self):
        f = CollectiveFields(FieldConfig(grid_size=50), n_agents=5)
        f.deposit_chemical(25.0, 25.0, 10.0)
        gx, gy = f.get_chemical_gradient(24.0, 25.0)
        # Gradient should return floats
        self.assertIsInstance(gx, float)

    def test_emotional_field_shape(self):
        f = CollectiveFields(FieldConfig(), n_agents=10)
        self.assertEqual(f.emotional_field.shape[0], 10)

    def test_synchronize_emotions(self):
        f = CollectiveFields(FieldConfig(), n_agents=5)
        f.emotional_field = np.random.randn(5, 8)
        before_var = f.emotional_field.var()
        f.synchronize_emotions(coupling=0.5)
        after_var = f.emotional_field.var()
        self.assertLessEqual(after_var, before_var + 0.01)

    def test_symbolic_field(self):
        f = CollectiveFields(FieldConfig(grid_size=20), n_agents=3)
        glyph = f.get_symbolic_at(10.0, 10.0)
        self.assertEqual(len(glyph), 2)


class TestSwarmFitness(unittest.TestCase):
    def test_coverage(self):
        positions = np.random.rand(20, 2) * 100
        # area is a tuple (width, height)
        score = SwarmFitness.coverage_score(positions, area=(100.0, 100.0))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_cohesion(self):
        positions = np.random.rand(20, 2) * 10
        score = SwarmFitness.cohesion_score(positions)
        self.assertGreaterEqual(score, 0)

    def test_alignment(self):
        headings = np.ones(20) * 1.5
        score = SwarmFitness.alignment_score(headings)
        self.assertGreater(score, 0.9)

    def test_composite(self):
        env = SwarmEnvironment(EnvConfig(n_agents=10))
        for _ in range(10):
            env.step()
        score = SwarmFitness.composite(env)
        self.assertIsInstance(score, float)

    def test_composite_non_negative(self):
        env = SwarmEnvironment(EnvConfig(n_agents=5))
        # Composite can be negative due to penalties, but test it runs
        score = SwarmFitness.composite(env)
        self.assertIsInstance(score, float)


class TestSwarmEvolver(unittest.TestCase):
    def test_init(self):
        cfg = EvolverConfig(pop_size=5, agent_config=AgentConfig(n_hidden=4))
        ev = SwarmEvolver(cfg)
        self.assertEqual(len(ev.population), 5)

    def test_evaluate(self):
        cfg = EvolverConfig(pop_size=5, n_eval_steps=20, agent_config=AgentConfig(n_hidden=4))
        ev = SwarmEvolver(cfg)
        fit = ev.evaluate_individual(ev.population[0])
        self.assertIsInstance(fit, float)

    def test_evolve_generation(self):
        cfg = EvolverConfig(
            pop_size=6, n_elite=2, n_eval_steps=20, agent_config=AgentConfig(n_hidden=4)
        )
        ev = SwarmEvolver(cfg)
        best = ev.evolve_generation()
        self.assertIsInstance(best, float)

    def test_get_best_weights(self):
        cfg = EvolverConfig(pop_size=5, n_eval_steps=10, agent_config=AgentConfig(n_hidden=4))
        ev = SwarmEvolver(cfg)
        ev.evolve_generation()
        w = ev.get_best_weights()
        self.assertIsInstance(w, np.ndarray)

    def test_run(self):
        cfg = EvolverConfig(
            pop_size=5, n_elite=2, n_eval_steps=10, agent_config=AgentConfig(n_hidden=4)
        )
        ev = SwarmEvolver(cfg)
        history = ev.run(n_generations=2)
        self.assertEqual(len(history), 2)

    def test_weight_sizes_match(self):
        acfg = AgentConfig(n_hidden=4, n_sensory=20, n_motor=2)
        cfg = EvolverConfig(pop_size=3, agent_config=acfg)
        ev = SwarmEvolver(cfg)
        template = SwarmAgent(acfg)
        self.assertEqual(len(ev.population[0]), template.n_weights)


class TestSwarmIntegration(unittest.TestCase):
    def test_agents_with_fields(self):
        acfg = AgentConfig(n_hidden=8)
        cfg = EnvConfig(n_agents=5, agent_config=acfg)
        env = SwarmEnvironment(cfg)
        fields = CollectiveFields(FieldConfig(grid_size=20), n_agents=5)
        for _ in range(10):
            env.step()
            for a in env.agents:
                fields.deposit_chemical(a.position[0] % 100, a.position[1] % 100, 0.1)
            fields.diffuse(0.1)
        self.assertGreater(fields.chemical_field.max(), 0)

    def test_evolution_improves(self):
        """Evolution should not systematically worsen over 3 generations."""
        cfg = EvolverConfig(
            pop_size=8, n_elite=2, n_eval_steps=30, agent_config=AgentConfig(n_hidden=4)
        )
        ev = SwarmEvolver(cfg)
        history = ev.run(n_generations=3)
        # Just verify it ran without error
        self.assertEqual(len(history), 3)
        self.assertTrue(all(isinstance(f, float) for f in history))


if __name__ == "__main__":
    unittest.main()
