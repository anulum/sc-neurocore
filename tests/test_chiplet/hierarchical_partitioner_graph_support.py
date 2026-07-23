# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_hierarchical_partitioner_graph.py

from __future__ import annotations

"""Graph cache, seed allocation, bisection, and refinement contracts."""
import numpy as np
import pytest
from sc_neurocore.chiplet import (
    CorrelationAwareGraph,
    CorrelationEdge,
    HierarchicalPartitioner,
    LFSRSeedAllocator,
    calculate_boundary_scc,
    calculate_edge_cut,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    build_graph as _build_graph,
    make_biclique as _make_biclique,
    make_chain_graph as _make_chain_graph,
)

__all__ = ['np', 'pytest', 'CorrelationAwareGraph', 'CorrelationEdge', 'HierarchicalPartitioner', 'LFSRSeedAllocator', 'calculate_boundary_scc', 'calculate_edge_cut', '_build_graph', '_make_biclique', '_make_chain_graph']
