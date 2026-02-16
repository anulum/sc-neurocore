"""Tests for SwarmAgent."""

import numpy as np
import pytest

from sc_neurocore.swarm.agent import SwarmAgent, AgentConfig


class TestSwarmAgent:
    def test_init_default(self):
        agent = SwarmAgent(agent_id=0, seed=42)
        assert agent.agent_id == 0
        assert agent.W_in.shape == (16, 20)
        assert agent.W_rec.shape == (16, 16)
        assert agent.W_out.shape == (2, 16)

    def test_custom_config(self):
        cfg = AgentConfig(n_sensory=10, n_hidden=8, n_motor=3)
        agent = SwarmAgent(0, config=cfg, seed=0)
        assert agent.W_in.shape == (8, 10)
        assert agent.W_out.shape == (3, 8)

    def test_sense_output_shape(self):
        agent = SwarmAgent(0, seed=1)
        sensory = agent.sense(
            neighbor_dists=np.ones(8),
            obstacle_dists=np.ones(3),
            target_dists=np.ones(2),
            chem_gradient=np.array([0.5, 0.5]),
            symbolic_value=np.array([0.3, 0.7]),
        )
        assert sensory.shape == (20,)
        assert np.all(sensory >= 0) and np.all(sensory <= 1)

    def test_sense_fewer_neighbors(self):
        agent = SwarmAgent(0, seed=2)
        sensory = agent.sense(
            neighbor_dists=np.array([0.5, 0.3]),  # Only 2 neighbors
            obstacle_dists=np.array([0.8]),        # Only 1 obstacle
            target_dists=np.array([0.4]),           # Only 1 target
            chem_gradient=np.array([0.5, 0.5]),
            symbolic_value=np.array([0.5, 0.5]),
        )
        assert sensory.shape == (20,)
        # Unfilled slots should be 1.0 (far away)
        assert sensory[2] == 1.0  # 3rd neighbor slot

    def test_think_output_shape(self):
        agent = SwarmAgent(0, seed=3)
        sensory = np.random.rand(20)
        motor = agent.think(sensory)
        assert motor.shape == (2,)
        assert motor[0] >= 0  # speed non-negative
        assert abs(motor[1]) <= agent.config.max_turn

    def test_act_updates_position(self):
        agent = SwarmAgent(0, seed=4)
        agent.x, agent.y, agent.heading = 50.0, 50.0, 0.0
        agent.act(speed=1.0, turn_angle=0.0)
        assert agent.x == pytest.approx(51.0)
        assert agent.y == pytest.approx(50.0)

    def test_act_turn(self):
        agent = SwarmAgent(0, seed=5)
        agent.x, agent.y, agent.heading = 50.0, 50.0, 0.0
        agent.act(speed=1.0, turn_angle=np.pi / 2)
        assert agent.heading == pytest.approx(np.pi / 2)
        assert agent.y > 50.0  # Moved upward

    def test_weights_property(self):
        agent = SwarmAgent(0, seed=6)
        w = agent.weights
        expected_size = 16 * 20 + 16 * 16 + 2 * 16  # W_in + W_rec + W_out
        assert len(w) == expected_size

    def test_weights_setter(self):
        agent = SwarmAgent(0, seed=7)
        new_w = np.random.rand(len(agent.weights))
        agent.weights = new_w
        np.testing.assert_allclose(agent.weights, new_w)

    def test_clone(self):
        agent = SwarmAgent(0, seed=8)
        agent.think(np.random.rand(20))  # Build some state
        clone = agent.clone(new_id=99)
        assert clone.agent_id == 99
        np.testing.assert_allclose(clone.weights, agent.weights)

    def test_reset_neural_state(self):
        agent = SwarmAgent(0, seed=9)
        agent.think(np.random.rand(20))
        agent.reset_neural_state()
        assert np.all(agent.membrane == 0)
        assert np.all(agent.firing_rates == 0)

    def test_get_neural_state(self):
        agent = SwarmAgent(0, seed=10)
        state = agent.get_neural_state()
        assert "agent_id" in state
        assert "position" in state
        assert "mean_activity" in state

    def test_deterministic_with_seed(self):
        a1 = SwarmAgent(0, seed=42)
        a2 = SwarmAgent(0, seed=42)
        np.testing.assert_allclose(a1.W_in, a2.W_in)
        np.testing.assert_allclose(a1.W_rec, a2.W_rec)
