# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmEnvironment from former test_swarm_fitness_contracts.py

"""Focused suite: TestSwarmEnvironment from former test_swarm_fitness_contracts.py."""

from __future__ import annotations

from tests.swarm_fitness_contracts_support import *  # noqa: F403


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
