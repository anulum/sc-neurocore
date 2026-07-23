# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPreExistingEdgeCases from former test_hierarchical_partitioner_core.py

"""Focused suite: TestPreExistingEdgeCases from former test_hierarchical_partitioner_core.py."""

from __future__ import annotations

from hierarchical_partitioner_core_support import *  # noqa: F403

class TestPreExistingEdgeCases:
    """Two pre-existing edge-case lines (calculate_imbalance_ratio
    `ideal == 0` and MigrationPlanner `recs >= max_recommendations`)
    were uncovered by the original suite. Pin them so the chiplet
    package reaches 100 % coverage."""

    def test_imbalance_ratio_with_zero_ideal(self) -> None:
        from sc_neurocore.chiplet.hierarchical_partitioner import (
            calculate_imbalance_ratio,
        )

        # Empty partition list → ideal=0/0 short-circuits at line 895 (`not sizes`),
        # but [empty, empty] gives total=0, ideal=0 → triggers line 899/900.
        result = calculate_imbalance_ratio([[], []])
        assert result == 0.0

    def test_load_balancer_respects_max_recommendations(self) -> None:
        from sc_neurocore.chiplet.hierarchical_partitioner import (
            CorrelationLoadBalancer,
        )

        # Strong imbalance + many cross-partition edges → planner
        # produces multiple candidates; cap at 1 → forces the
        # `len(recs) >= max_recommendations` break (line 1116).
        edges = [CorrelationEdge(u=v, v=20, conn_weight=1.0, scc_weight=0.5) for v in range(20)]
        g = CorrelationAwareGraph(num_vertices=21, edges=edges)
        partitions = [list(range(20)), [20]]
        planner = CorrelationLoadBalancer(imbalance_threshold=0.05)
        recs = planner.recommend_migrations(
            g,
            partitions,
            max_recommendations=1,
        )
        assert len(recs) <= 1
