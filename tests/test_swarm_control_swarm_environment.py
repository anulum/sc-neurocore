# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmEnvironment from former test_swarm_control.py

"""Focused suite: TestSwarmEnvironment from former test_swarm_control.py."""

from __future__ import annotations

from tests.swarm_control_support import *  # noqa: F403

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
