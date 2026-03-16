# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Coverage tests for swarm.swarm_env, swarm.collective_fields, swarm.fitness."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.swarm.swarm_env import SwarmEnvironment, EnvConfig
from sc_neurocore.swarm.collective_fields import (
    CollectiveFields,
    FieldConfig,
    _apply_laplacian,
)
from sc_neurocore.swarm.fitness import SwarmFitness


class TestSwarmEnvironment:
    @pytest.fixture()
    def env(self):
        return SwarmEnvironment(EnvConfig(n_agents=5, n_obstacles=2, n_targets=2, seed=42))

    def test_init(self, env):
        assert len(env.agents) == 5
        assert env.obstacles.shape == (2, 3)
        assert env.targets.shape == (2, 2)

    def test_get_positions(self, env):
        pos = env.get_positions()
        assert pos.shape == (5, 2)

    def test_get_headings(self, env):
        h = env.get_headings()
        assert h.shape == (5,)

    def test_pairwise_distances(self, env):
        d = env.get_pairwise_distances()
        assert d.shape == (5, 5)
        assert np.allclose(np.diag(d), 0)

    def test_neighbor_distances(self, env):
        nd = env.get_neighbor_distances(0, k=3)
        assert nd.shape == (3,)

    def test_obstacle_distances(self, env):
        od = env.get_obstacle_distances(0, k=2)
        assert od.shape == (2,)

    def test_target_distances(self, env):
        td = env.get_target_distances(0, k=2)
        assert td.shape == (2,)

    def test_step_no_fields(self, env):
        env.step(dt=1.0)
        assert env.step_count == 1

    def test_step_with_fields(self, env):
        fields = CollectiveFields(
            FieldConfig(seed=42),
            env_width=env.cfg.width,
            env_height=env.cfg.height,
            n_agents=env.cfg.n_agents,
        )
        env.step(dt=1.0, fields=fields)
        assert env.step_count == 1

    def test_boundary_clamp(self):
        cfg = EnvConfig(n_agents=2, n_obstacles=0, n_targets=0, boundary_mode="clamp", seed=1)
        env = SwarmEnvironment(cfg)
        env.agents[0].position = np.array([-5.0, -5.0])
        env._apply_boundary(env.agents[0])
        assert env.agents[0].position[0] >= 0
        assert env.agents[0].position[1] >= 0

    def test_get_state(self, env):
        state = env.get_state()
        assert "positions" in state
        assert "targets_captured" in state
        assert state["step"] == 0


class TestCollectiveFields:
    @pytest.fixture()
    def fields(self):
        return CollectiveFields(FieldConfig(grid_size=10, seed=42), env_width=100, env_height=100)

    def test_init_shapes(self, fields):
        assert fields.chemical_field.shape == (10, 10)
        assert fields.emotional_field.shape == (20, 8)
        assert fields.symbolic_field.shape == (10, 10, 2)

    def test_deposit_chemical(self, fields):
        fields.deposit_chemical(50.0, 50.0, 1.0)
        assert fields.chemical_field.sum() > 0

    def test_deposit_chemical_negative_ignored(self, fields):
        fields.deposit_chemical(50.0, 50.0, -1.0)
        assert fields.chemical_field.sum() == 0

    def test_diffuse(self, fields):
        fields.deposit_chemical(50.0, 50.0, 10.0)
        before = fields.chemical_field.copy()
        fields.diffuse(dt=1.0)
        assert not np.array_equal(before, fields.chemical_field)

    def test_get_chemical_gradient(self, fields):
        fields.deposit_chemical(60.0, 50.0, 10.0)
        gx, gy = fields.get_chemical_gradient(50.0, 50.0)
        assert isinstance(gx, float)

    def test_synchronize_emotions(self, fields):
        fields.emotional_field[0] = 1.0
        fields.synchronize_emotions()
        assert fields.emotional_field[0, 0] < 1.0

    def test_synchronize_emotions_custom_coupling(self, fields):
        fields.emotional_field[0] = 1.0
        fields.synchronize_emotions(coupling=0.5)
        assert fields.emotional_field[0, 0] < 1.0

    def test_symbolic_deposit_and_read(self, fields):
        fields.deposit_symbolic(25.0, 25.0, 0, 5.0)
        val = fields.get_symbolic_at(25.0, 25.0)
        assert val[0] == 5.0

    def test_apply_laplacian(self):
        field = np.zeros((5, 5))
        field[2, 2] = 1.0
        lap = _apply_laplacian(field)
        assert lap[2, 2] < 0
        assert lap[1, 2] > 0


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
