# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmFitness from former test_swarm_fitness_contracts.py

"""Focused suite: TestSwarmFitness from former test_swarm_fitness_contracts.py."""

from __future__ import annotations

from tests.swarm_fitness_contracts_support import *  # noqa: F403


class TestSwarmFitness:
    def test_coverage_score(self):
        pos = np.array([[10, 10], [50, 50], [90, 90]], dtype=float)
        score = SwarmFitness.coverage_score(pos, (100, 100))
        assert 0 < score <= 1

    def test_cohesion_score(self):
        pos = np.array([[10, 10], [12, 12], [14, 14]], dtype=float)
        score = SwarmFitness.cohesion_score(pos)
        assert 0 <= score <= 1

    def test_cohesion_single_agent(self):
        assert SwarmFitness.cohesion_score(np.array([[0, 0]], dtype=float)) == 0.0

    def test_alignment_score(self):
        headings = np.array([0.0, 0.0, 0.0])
        assert SwarmFitness.alignment_score(headings) == pytest.approx(1.0, abs=0.01)

    def test_alignment_empty(self):
        assert SwarmFitness.alignment_score(np.array([])) == 0.0

    def test_target_score(self):
        pos = np.array([[10, 10], [20, 20]], dtype=float)
        targets = np.array([[10, 10]], dtype=float)
        score = SwarmFitness.target_score(pos, targets)
        assert score > 0.5

    def test_target_score_empty(self):
        assert SwarmFitness.target_score(np.array([[0, 0]], dtype=float), np.empty((0, 2))) == 0.0

    def test_obstacle_penalty(self):
        pos = np.array([[10, 10]], dtype=float)
        obs = np.array([[10, 10, 5]], dtype=float)
        penalty = SwarmFitness.obstacle_penalty(pos, obs)
        assert penalty == 1.0

    def test_obstacle_penalty_empty(self):
        assert (
            SwarmFitness.obstacle_penalty(np.array([[0, 0]], dtype=float), np.empty((0, 3))) == 0.0
        )

    def test_composite(self):
        env = SwarmEnvironment(EnvConfig(n_agents=5, seed=42))
        score = SwarmFitness.composite(env)
        assert isinstance(score, float)
