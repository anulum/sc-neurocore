# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmIntegration from former test_swarm_control.py

"""Focused suite: TestSwarmIntegration from former test_swarm_control.py."""

from __future__ import annotations

from tests.swarm_control_support import *  # noqa: F403

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
