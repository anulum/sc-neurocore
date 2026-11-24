# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmEvolver from former test_swarm_control.py

"""Focused suite: TestSwarmEvolver from former test_swarm_control.py."""

from __future__ import annotations

import numpy as np
import unittest
from sc_neurocore.swarm import (
    AgentConfig,
    EvolverConfig,
    SwarmAgent,
    SwarmEvolver,
)


class TestSwarmEvolver(unittest.TestCase):
    """Neuroevolution of swarm-agent weights."""

    def test_init(self) -> None:
        """A population size of five gives five individuals."""
        cfg = EvolverConfig(pop_size=5, agent_config=AgentConfig(n_hidden=4))
        ev = SwarmEvolver(cfg)
        self.assertEqual(len(ev.population), 5)

    def test_evaluate(self) -> None:
        """Evaluating one individual gives a float fitness."""
        cfg = EvolverConfig(pop_size=5, n_eval_steps=20, agent_config=AgentConfig(n_hidden=4))
        ev = SwarmEvolver(cfg)
        fit = ev.evaluate_individual(ev.population[0])
        self.assertIsInstance(fit, float)

    def test_evolve_generation(self) -> None:
        """One generation gives a float best fitness."""
        cfg = EvolverConfig(
            pop_size=6, n_elite=2, n_eval_steps=20, agent_config=AgentConfig(n_hidden=4)
        )
        ev = SwarmEvolver(cfg)
        best = ev.evolve_generation()
        self.assertIsInstance(best, float)

    def test_get_best_weights(self) -> None:
        """The best weights after a generation are an array."""
        cfg = EvolverConfig(pop_size=5, n_eval_steps=10, agent_config=AgentConfig(n_hidden=4))
        ev = SwarmEvolver(cfg)
        ev.evolve_generation()
        w = ev.get_best_weights()
        self.assertIsInstance(w, np.ndarray)

    def test_run(self) -> None:
        """Two generations give a history of two entries."""
        cfg = EvolverConfig(
            pop_size=5, n_elite=2, n_eval_steps=10, agent_config=AgentConfig(n_hidden=4)
        )
        ev = SwarmEvolver(cfg)
        history = ev.run(n_generations=2)
        self.assertEqual(len(history), 2)

    def test_weight_sizes_match(self) -> None:
        """Each individual has as many weights as an agent of the same configuration."""
        acfg = AgentConfig(n_hidden=4, n_sensory=20, n_motor=2)
        cfg = EvolverConfig(pop_size=3, agent_config=acfg)
        ev = SwarmEvolver(cfg)
        template = SwarmAgent(acfg)
        self.assertEqual(len(ev.population[0]), template.n_weights)
