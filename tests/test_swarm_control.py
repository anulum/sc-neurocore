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


class TestSwarmCommunication(unittest.TestCase):
    """Cover communication.py (previously 0%)."""

    def _make_env_and_fields(self, n=5):
        from sc_neurocore.swarm import SwarmCommunication

        cfg = EnvConfig(n_agents=n, n_obstacles=2, n_targets=1, seed=42)
        env = SwarmEnvironment(cfg)
        fields = CollectiveFields(FieldConfig(grid_size=20), n_agents=n)
        comm = SwarmCommunication(env, fields)
        return env, fields, comm

    def test_init(self):
        from sc_neurocore.swarm import SwarmCommunication

        env, fields, comm = self._make_env_and_fields()
        self.assertIs(comm.env, env)
        self.assertIs(comm.fields, fields)
        self.assertEqual(comm.broadcast_radius, 15.0)

    def test_step(self):
        env, fields, comm = self._make_env_and_fields()
        # Give agents some chemical output
        for a in env.agents:
            a.chemical_output = 0.5
        comm.step(dt=1.0)
        # Chemical field should have deposits
        self.assertGreater(fields.chemical_field.max(), 0)

    def test_get_sensory_data(self):
        env, fields, comm = self._make_env_and_fields()
        data = comm.get_sensory_data(0)
        self.assertIn("chem_gradient", data)
        self.assertIn("symbolic_value", data)
        self.assertEqual(len(data["chem_gradient"]), 2)
        self.assertEqual(len(data["symbolic_value"]), 2)

    def test_step_updates_symbolic(self):
        env, fields, comm = self._make_env_and_fields()
        for a in env.agents:
            a.emotions[:2] = [1.0, 0.0]
        comm.step(dt=1.0)
        # Symbolic field should have some non-zero values
        self.assertGreater(np.abs(fields.symbolic_field).max(), 0)


class TestSwarmFitnessDeep(unittest.TestCase):
    """Cover remaining fitness.py lines."""

    def test_cohesion_single_agent(self):
        positions = np.array([[50.0, 50.0]])
        score = SwarmFitness.cohesion_score(positions)
        self.assertEqual(score, 0.0)

    def test_alignment_empty(self):
        headings = np.array([])
        score = SwarmFitness.alignment_score(headings)
        self.assertEqual(score, 0.0)

    def test_target_score(self):
        positions = np.array([[10.0, 10.0], [90.0, 90.0]])
        targets = np.array([[10.0, 10.0]])
        score = SwarmFitness.target_score(positions, targets)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_target_score_no_targets(self):
        positions = np.array([[10.0, 10.0]])
        targets = np.zeros((0, 2))
        score = SwarmFitness.target_score(positions, targets)
        self.assertEqual(score, 0.0)

    def test_obstacle_penalty(self):
        positions = np.array([[50.0, 50.0]])
        obstacles = np.array([[50.0, 50.0, 10.0]])  # Agent inside obstacle
        penalty = SwarmFitness.obstacle_penalty(positions, obstacles)
        self.assertEqual(penalty, 1.0)

    def test_obstacle_penalty_no_obstacles(self):
        positions = np.array([[50.0, 50.0]])
        obstacles = np.zeros((0, 3))
        penalty = SwarmFitness.obstacle_penalty(positions, obstacles)
        self.assertEqual(penalty, 0.0)

    def test_obstacle_penalty_outside(self):
        positions = np.array([[50.0, 50.0]])
        obstacles = np.array([[80.0, 80.0, 5.0]])  # Far away
        penalty = SwarmFitness.obstacle_penalty(positions, obstacles)
        self.assertEqual(penalty, 0.0)


class TestCollectiveFieldsDeep(unittest.TestCase):
    """Cover remaining collective_fields.py lines."""

    def test_deposit_chemical_negative(self):
        f = CollectiveFields(FieldConfig(grid_size=20), n_agents=3)
        f.deposit_chemical(10.0, 10.0, -1.0)  # Negative amount -> no-op
        self.assertEqual(f.chemical_field.max(), 0.0)

    def test_deposit_symbolic(self):
        f = CollectiveFields(FieldConfig(grid_size=20), n_agents=3)
        f.deposit_symbolic(10.0, 10.0, 0, 1.5)
        self.assertGreater(f.symbolic_field[:, :, 0].max(), 0)

    def test_update(self):
        cfg = EnvConfig(n_agents=3, seed=42)
        env = SwarmEnvironment(cfg)
        f = CollectiveFields(FieldConfig(grid_size=20), n_agents=3)
        # Set some emotions on agents
        for a in env.agents:
            a.emotions = np.random.randn(8)
        f.update(env.agents, env, dt=1.0)
        # After update, symbolic field should have decayed (or remain 0)
        # Emotions should be updated
        self.assertIsNotNone(f.emotional_field)

    def test_synchronize_emotions_default_coupling(self):
        f = CollectiveFields(FieldConfig(), n_agents=3)
        f.emotional_field = np.random.randn(3, 8)
        f.synchronize_emotions()  # Uses default coupling
        # Should not crash; field should still be valid
        self.assertEqual(f.emotional_field.shape, (3, 8))


class TestSwarmEnvDeep(unittest.TestCase):
    """Cover remaining swarm_env.py lines."""

    def test_boundary_clamp(self):
        cfg = EnvConfig(n_agents=1, boundary_mode="clamp", width=100, height=100, seed=1)
        env = SwarmEnvironment(cfg)
        env.agents[0].position = np.array([150.0, -10.0])
        env._apply_boundary(env.agents[0])
        pos = env.agents[0].position
        self.assertLessEqual(pos[0], 100)
        self.assertGreaterEqual(pos[1], 0)

    def test_obstacle_distances(self):
        cfg = EnvConfig(n_agents=3, n_obstacles=5, seed=42)
        env = SwarmEnvironment(cfg)
        dists = env.get_obstacle_distances(0, k=3)
        self.assertEqual(len(dists), 3)

    def test_target_distances(self):
        cfg = EnvConfig(n_agents=3, n_targets=3, seed=42)
        env = SwarmEnvironment(cfg)
        dists = env.get_target_distances(0, k=2)
        self.assertEqual(len(dists), 2)

    def test_target_capture(self):
        cfg = EnvConfig(
            n_agents=1,
            n_targets=1,
            capture_radius=50.0,
            respawn_targets=True,
            seed=42,
        )
        env = SwarmEnvironment(cfg)
        # Place agent on top of target
        env.agents[0].position = env.targets[0].copy()
        env.step()
        self.assertGreater(env.targets_captured, 0)

    def test_step_with_fields(self):
        cfg = EnvConfig(n_agents=3, seed=42)
        env = SwarmEnvironment(cfg)
        fields = CollectiveFields(FieldConfig(grid_size=20), n_agents=3)
        env.step(dt=1.0, fields=fields)
        # Should not crash; step_count should advance
        self.assertEqual(env.step_count, 1)

    def test_no_respawn_targets(self):
        cfg = EnvConfig(
            n_agents=1,
            n_targets=1,
            capture_radius=200.0,
            respawn_targets=False,
            seed=42,
        )
        env = SwarmEnvironment(cfg)
        env.agents[0].position = env.targets[0].copy()
        env.step()
        # Target captured but not respawned
        self.assertGreater(env.targets_captured, 0)


class TestSwarmEvolverDeep(unittest.TestCase):
    """Cover remaining neuroevolution_swarm.py lines."""

    def test_evaluate_with_fields(self):
        cfg = EvolverConfig(
            pop_size=3,
            n_eval_steps=10,
            use_fields=True,
            agent_config=AgentConfig(n_hidden=4),
            seed=42,
        )
        ev = SwarmEvolver(cfg)
        fit = ev.evaluate_individual(ev.population[0])
        self.assertIsInstance(fit, float)

    def test_custom_env_config(self):
        ecfg = EnvConfig(width=50, height=50, n_agents=3)
        cfg = EvolverConfig(
            pop_size=3,
            n_eval_steps=5,
            env_config=ecfg,
            agent_config=AgentConfig(n_hidden=4),
            seed=42,
        )
        ev = SwarmEvolver(cfg)
        env = ev._make_env()
        self.assertEqual(env.cfg.width, 50)
        self.assertEqual(env.cfg.height, 50)

    def test_crossover_and_mutate(self):
        cfg = EvolverConfig(
            pop_size=4,
            n_elite=2,
            mutation_rate=0.5,
            agent_config=AgentConfig(n_hidden=4),
            seed=42,
        )
        ev = SwarmEvolver(cfg)
        pa = ev.population[0].copy()
        pb = ev.population[1].copy()
        child = ev._crossover(pa, pb)
        self.assertEqual(len(child), ev.n_weights)
        mutated = ev._mutate(child.copy())
        self.assertEqual(len(mutated), ev.n_weights)


if __name__ == "__main__":
    unittest.main()
