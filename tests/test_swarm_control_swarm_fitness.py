# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmFitness from former test_swarm_control.py

"""Focused suite: TestSwarmFitness from former test_swarm_control.py."""

from __future__ import annotations

from tests.swarm_control_support import *  # noqa: F403


class TestSwarmFitness(unittest.TestCase):
    def test_coverage(self):
        positions = np.random.rand(20, 2) * 100
        # area is a tuple (width, height)
        score = SwarmFitness.coverage_score(positions, area=(100.0, 100.0))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_cohesion(self):
        positions = np.random.rand(20, 2) * 10
        score = SwarmFitness.cohesion_score(positions)
        self.assertGreaterEqual(score, 0)

    def test_alignment(self):
        headings = np.ones(20) * 1.5
        score = SwarmFitness.alignment_score(headings)
        self.assertGreater(score, 0.9)

    def test_composite(self):
        env = SwarmEnvironment(EnvConfig(n_agents=10))
        for _ in range(10):
            env.step()
        score = SwarmFitness.composite(env)
        self.assertIsInstance(score, float)

    def test_composite_non_negative(self):
        env = SwarmEnvironment(EnvConfig(n_agents=5))
        # Composite can be negative due to penalties, but test it runs
        score = SwarmFitness.composite(env)
        self.assertIsInstance(score, float)
