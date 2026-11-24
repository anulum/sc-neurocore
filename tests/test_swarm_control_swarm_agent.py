# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmAgent from former test_swarm_control.py

"""Focused suite: TestSwarmAgent from former test_swarm_control.py."""

from __future__ import annotations

import numpy as np
import unittest
from sc_neurocore.swarm import (
    AgentConfig,
    SwarmAgent,
)


class TestSwarmAgent(unittest.TestCase):
    """The spiking swarm agent: its network, weights and motion."""

    def test_init(self) -> None:
        """A default agent has id 0 and a 2-D position."""
        a = SwarmAgent(AgentConfig())
        self.assertEqual(a.agent_id, 0)
        self.assertEqual(a.position.shape, (2,))

    def test_weights_setter_rejects_wrong_size(self) -> None:
        """Setting a weight vector of the wrong size raises ValueError."""
        a = SwarmAgent(AgentConfig())
        with self.assertRaises(ValueError):
            a.weights = np.zeros(3, dtype=np.float64)

    def test_init_config(self) -> None:
        """Twenty sensory inputs and eight hidden units give an 8x20 input matrix."""
        cfg = AgentConfig(n_sensory=20, n_hidden=8, n_motor=2)
        a = SwarmAgent(cfg)
        self.assertEqual(a.W_in.shape, (8, 20))

    def test_think(self) -> None:
        """Thinking on zero input gives a float speed and a float turn."""
        a = SwarmAgent(AgentConfig())
        sensory = np.zeros(a.cfg.n_sensory)
        speed, turn = a.think(sensory)
        self.assertIsInstance(speed, float)
        self.assertIsInstance(turn, float)

    def test_act(self) -> None:
        """Acting with unit speed moves the agent."""
        a = SwarmAgent(AgentConfig())
        pos_before = a.position.copy()
        a.act(1.0, 0.0)
        self.assertFalse(np.allclose(a.position, pos_before))

    def test_weights_property(self) -> None:
        """The flat weight vector holds the input, recurrent and output weights."""
        cfg = AgentConfig(n_hidden=8, n_sensory=20, n_motor=2)
        a = SwarmAgent(cfg)
        w = a.weights
        n_expected = 8 * 20 + 8 * 8 + 2 * 8  # W_in + W_rec + W_out
        self.assertEqual(len(w), n_expected)

    def test_weights_setter(self) -> None:
        """A flat weight vector of the right size is stored."""
        cfg = AgentConfig(n_hidden=8)
        a = SwarmAgent(cfg)
        w = np.ones(a.n_weights) * 0.5
        a.weights = w
        self.assertTrue(np.allclose(a.weights, 0.5))

    def test_reset(self) -> None:
        """A reset after acting zeroes the membrane."""
        a = SwarmAgent(AgentConfig())
        a.act(1.0, 0.5)
        a.reset()
        self.assertTrue(np.allclose(a.membrane, 0))

    def test_neural_state(self) -> None:
        """Thinking leaves a firing rate and a membrane state."""
        a = SwarmAgent(AgentConfig())
        a.think(np.zeros(a.cfg.n_sensory))
        self.assertIsNotNone(a.firing_rate)
        self.assertIsNotNone(a.membrane)

    def test_heading_wraps(self) -> None:
        """A large turn leaves the heading inside [0, 2 pi)."""
        a = SwarmAgent(AgentConfig())
        a.heading = 0
        a.act(0.0, 100.0)  # Large turn
        self.assertGreaterEqual(a.heading, 0)
        self.assertLess(a.heading, 2 * np.pi)
