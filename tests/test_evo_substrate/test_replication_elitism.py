# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestElitism from former test_replication.py

"""Focused suite: TestElitism from former test_replication.py."""

from __future__ import annotations

from tests.test_evo_substrate.replication_support import *  # noqa: F403

class TestElitism:
    def test_best_survives_cull(self) -> None:
        re = ReplicationEngine(max_population=20, elitism=1)
        for i in range(10):
            g = Genome()
            g.topology.num_neurons = 10 + i * 10
            re.seed(g)
        re.evaluate_all(lambda g: {"accuracy": g.topology.num_neurons / 200.0})
        assert re.best_organism is not None
        best_id = re.best_organism.genome.genome_id
        re.select_and_cull(survival_fraction=0.3)
        remaining_ids = [o.genome.genome_id for o in re.population]
        assert best_id in remaining_ids

    def test_diversity_in_evolve_result(self) -> None:
        re = ReplicationEngine(max_population=8)
        for _ in range(4):
            re.seed(Genome())
        result = re.evolve_generation(lambda g: {"accuracy": 0.5})
        assert "diversity" in result
