# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_hierarchical_partitioner_core.py

from __future__ import annotations

"""Edge cases, backend validation, and coarse performance regression gates."""
import time
import numpy as np
import pytest
from sc_neurocore.chiplet import (
    CorrelationAwareGraph,
    CorrelationEdge,
    HierarchicalPartitioner,
)
from tests.test_chiplet.hierarchical_partitioner_support import build_graph as _build_graph
_TIMING_REPEATS = 5
def _min_partition_ms(n_vertices: int) -> float:
    """Return the minimum ``HierarchicalPartitioner.partition`` wall-clock in ms.

    The minimum over ``_TIMING_REPEATS`` runs is the sample least contaminated by
    shared-runner preemption, so a single noisy-neighbour spike cannot inflate a
    wall-clock timing assertion, while a genuine super-linear regression still
    shows in every repeat and is caught. A warm-up run first pays the one-time
    process/JIT cost so only steady-state partition compute is measured.
    """
    partitioner = HierarchicalPartitioner(num_partitions=4)
    partitioner.partition(_build_graph(50, seed=7))  # warm process / JIT / imports
    best_ms = float("inf")
    for _ in range(_TIMING_REPEATS):
        graph = _build_graph(n_vertices, avg_degree=8, seed=42)
        start = time.perf_counter()
        partitioner.partition(graph)
        best_ms = min(best_ms, (time.perf_counter() - start) * 1000.0)
    return best_ms

__all__ = ['time', 'np', 'pytest', 'CorrelationAwareGraph', 'CorrelationEdge', 'HierarchicalPartitioner', '_build_graph', '_TIMING_REPEATS', '_min_partition_ms']
