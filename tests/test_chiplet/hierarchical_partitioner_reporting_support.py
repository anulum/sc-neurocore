# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_hierarchical_partitioner_reporting.py

from __future__ import annotations

"""Boundary, balancing, rank mapping, repartitioning, and report contracts."""
from sc_neurocore.chiplet import (
    BoundarySyncConfig,
    BoundarySyncProtocol,
    CorrelationAwareGraph,
    CorrelationEdge,
    CorrelationLoadBalancer,
    GhostCellManager,
    HierarchicalPartitioner,
    HierarchyLevel,
    RankMapper,
    build_partition_report,
    calculate_edge_cut,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    make_chain_graph as _make_chain_graph,
)

__all__ = [
    "BoundarySyncConfig",
    "BoundarySyncProtocol",
    "CorrelationAwareGraph",
    "CorrelationEdge",
    "CorrelationLoadBalancer",
    "GhostCellManager",
    "HierarchicalPartitioner",
    "HierarchyLevel",
    "RankMapper",
    "build_partition_report",
    "calculate_edge_cut",
    "_make_chain_graph",
]
