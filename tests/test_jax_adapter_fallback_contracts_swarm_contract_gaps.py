# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmContractGaps from former test_jax_adapter_fallback_contracts.py

"""Focused suite: TestSwarmContractGaps from former test_jax_adapter_fallback_contracts.py."""

from __future__ import annotations

import numpy as np


class TestSwarmContractGaps:
    """Remaining swarm fitness and environment paths."""

    def test_fitness_cohesion_single_agent(self) -> None:
        """A single agent has its cohesion score."""
        from sc_neurocore.swarm.fitness import SwarmFitness

        assert SwarmFitness.cohesion_score(np.array([[0.0, 0.0]])) == 0.0

    def test_fitness_alignment_empty(self) -> None:
        """No agents give an alignment of 0."""
        from sc_neurocore.swarm.fitness import SwarmFitness

        assert SwarmFitness.alignment_score(np.array([])) == 0.0

    def test_fitness_target_no_targets(self) -> None:
        """No targets give a target score of 0."""
        from sc_neurocore.swarm.fitness import SwarmFitness

        pos = np.array([[1.0, 2.0]])
        assert SwarmFitness.target_score(pos, np.array([])) == 0.0

    def test_fitness_obstacle_no_obstacles(self) -> None:
        """No obstacles give no penalty."""
        from sc_neurocore.swarm.fitness import SwarmFitness

        pos = np.array([[1.0, 2.0]])
        assert SwarmFitness.obstacle_penalty(pos, np.array([])) == 0.0

    def test_collective_deposit_symbolic(self) -> None:
        """A symbolic deposit leaves a positive trace."""
        from sc_neurocore.swarm.collective_fields import CollectiveFields, FieldConfig

        fields = CollectiveFields(FieldConfig(grid_size=50))
        fields.deposit_symbolic(25.0, 25.0, 0, 1.0)
        val = fields.get_symbolic_at(25.0, 25.0)
        assert val[0] > 0

    def test_env_clamp_boundary(self) -> None:
        """Agents are clamped to the environment."""
        from sc_neurocore.swarm.swarm_env import SwarmEnvironment, EnvConfig
        from sc_neurocore.swarm.agent import SwarmAgent, AgentConfig

        env = SwarmEnvironment(EnvConfig(width=100, height=100, boundary_mode="clamp"))
        agent = SwarmAgent(AgentConfig(seed=42))
        agent.position = np.array([-10.0, 200.0])
        env._apply_boundary(agent)
        assert agent.position[0] >= 0
        assert agent.position[1] <= 100
