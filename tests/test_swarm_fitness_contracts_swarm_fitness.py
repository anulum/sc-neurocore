# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmFitness from former test_swarm_fitness_contracts.py

"""Focused suite: TestSwarmFitness from former test_swarm_fitness_contracts.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.swarm.fitness import (
    SwarmFitness,
)
from sc_neurocore.swarm.swarm_env import (
    EnvConfig,
    SwarmEnvironment,
)


class TestSwarmFitness:
    """Contracts of the swarm fitness scores."""

    def test_coverage_score(self) -> None:
        """Spread positions give a coverage inside (0, 1]."""
        pos = np.array([[10, 10], [50, 50], [90, 90]], dtype=float)
        score = SwarmFitness.coverage_score(pos, (100, 100))
        assert 0 < score <= 1

    def test_cohesion_score(self) -> None:
        """Clustered positions give a cohesion inside [0, 1]."""
        pos = np.array([[10, 10], [12, 12], [14, 14]], dtype=float)
        score = SwarmFitness.cohesion_score(pos)
        assert 0 <= score <= 1

    def test_cohesion_single_agent(self) -> None:
        """A single agent has a cohesion of zero."""
        assert SwarmFitness.cohesion_score(np.array([[0, 0]], dtype=float)) == 0.0

    def test_alignment_score(self) -> None:
        """Equal headings give an alignment of one."""
        headings = np.array([0.0, 0.0, 0.0])
        assert SwarmFitness.alignment_score(headings) == pytest.approx(1.0, abs=0.01)

    def test_alignment_empty(self) -> None:
        """No headings give an alignment of zero."""
        assert SwarmFitness.alignment_score(np.array([])) == 0.0

    def test_target_score(self) -> None:
        """An agent on a target gives a target score above 0.5."""
        pos = np.array([[10, 10], [20, 20]], dtype=float)
        targets = np.array([[10, 10]], dtype=float)
        score = SwarmFitness.target_score(pos, targets)
        assert score > 0.5

    def test_target_score_empty(self) -> None:
        """No targets give a target score of zero."""
        assert SwarmFitness.target_score(np.array([[0, 0]], dtype=float), np.empty((0, 2))) == 0.0

    def test_obstacle_penalty(self) -> None:
        """An agent inside an obstacle gives a penalty of one."""
        pos = np.array([[10, 10]], dtype=float)
        obs = np.array([[10, 10, 5]], dtype=float)
        penalty = SwarmFitness.obstacle_penalty(pos, obs)
        assert penalty == 1.0

    def test_obstacle_penalty_empty(self) -> None:
        """No obstacles give a penalty of zero."""
        assert (
            SwarmFitness.obstacle_penalty(np.array([[0, 0]], dtype=float), np.empty((0, 3))) == 0.0
        )

    def test_composite(self) -> None:
        """The composite fitness is a float."""
        env = SwarmEnvironment(EnvConfig(n_agents=5, seed=42))
        score = SwarmFitness.composite(env)
        assert isinstance(score, float)
