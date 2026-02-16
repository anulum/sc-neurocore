"""Tests for SwarmEnvironment."""

import numpy as np
import pytest

from sc_neurocore.swarm.swarm_env import SwarmEnvironment, EnvConfig


class TestSwarmEnvironment:
    def test_init_default(self):
        env = SwarmEnvironment()
        assert len(env.agents) == 20
        assert len(env.obstacles) == 5
        assert len(env.targets) == 3

    def test_custom_config(self):
        cfg = EnvConfig(n_agents=5, n_obstacles=2, n_targets=1, width=50, height=50)
        env = SwarmEnvironment(config=cfg)
        assert len(env.agents) == 5
        assert len(env.obstacles) == 2

    def test_agent_positions(self):
        env = SwarmEnvironment(EnvConfig(n_agents=10, seed=42))
        pos = env.get_agent_positions()
        assert pos.shape == (10, 2)
        assert np.all(pos >= 0)
        assert np.all(pos[:, 0] <= env.config.width)

    def test_pairwise_distances(self):
        env = SwarmEnvironment(EnvConfig(n_agents=5, seed=42))
        dists = env.get_pairwise_distances()
        assert dists.shape == (5, 5)
        np.testing.assert_allclose(np.diag(dists), 0.0)
        np.testing.assert_allclose(dists, dists.T)

    def test_neighbor_distances(self):
        env = SwarmEnvironment(EnvConfig(n_agents=10, seed=42))
        dists = env.get_neighbor_distances(0, k=4)
        assert len(dists) == 4
        assert np.all(dists >= 0) and np.all(dists <= 1)

    def test_obstacle_distances(self):
        env = SwarmEnvironment(EnvConfig(n_agents=5, n_obstacles=3, seed=42))
        dists = env.get_obstacle_distances(0, k=2)
        assert len(dists) == 2

    def test_target_distances(self):
        env = SwarmEnvironment(EnvConfig(n_agents=5, n_targets=2, seed=42))
        dists = env.get_target_distances(0, k=2)
        assert len(dists) == 2

    def test_step_advances_tick(self):
        env = SwarmEnvironment(EnvConfig(n_agents=3, seed=42))
        env.step()
        assert env.tick == 1

    def test_boundary_wrap(self):
        env = SwarmEnvironment(EnvConfig(width=100, height=100, n_agents=1, seed=42))
        env.agents[0].x = 105.0
        env.agents[0].y = -5.0
        env.step()
        assert 0 <= env.agents[0].x <= 100
        assert 0 <= env.agents[0].y <= 100

    def test_boundary_bounce(self):
        cfg = EnvConfig(width=100, height=100, n_agents=1, boundary_mode="bounce", seed=42)
        env = SwarmEnvironment(config=cfg)
        env.agents[0].x = -5.0
        env.agents[0].y = 50.0
        env.step()
        assert env.agents[0].x >= 0

    def test_reset(self):
        env = SwarmEnvironment(EnvConfig(n_agents=5, seed=42))
        env.step()
        env.step()
        env.reset()
        assert env.tick == 0

    def test_reset_keep_agents(self):
        env = SwarmEnvironment(EnvConfig(n_agents=3, seed=42))
        w0 = env.agents[0].weights.copy()
        env.reset(keep_agents=True)
        np.testing.assert_allclose(env.agents[0].weights, w0)

    def test_get_state(self):
        env = SwarmEnvironment(EnvConfig(n_agents=3, seed=42))
        state = env.get_state()
        assert "tick" in state
        assert "agent_positions" in state
        assert len(state["agent_positions"]) == 3
