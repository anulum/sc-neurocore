# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Historical hierarchical partitioner facade

"""Stable facade over responsibility-specific partitioning modules.

The split preserves established import identity, qualified names, and pickle
paths while graph modelling, backend dispatch, bisection, refinement, metrics,
boundary synchronisation, balancing, rank mapping, and reporting remain
independently auditable.
"""

from __future__ import annotations

from typing import Any

from sc_neurocore.chiplet import hierarchical_backend_runtime as _runtime
from sc_neurocore.chiplet.hierarchical_balancing import (
    CorrelationLoadBalancer,
    LoadMetrics,
    MigrationRecommendation,
    RankMapper,
)
from sc_neurocore.chiplet.hierarchical_boundary import (
    BoundarySyncConfig,
    BoundarySyncProtocol,
    GhostCellManager,
)
from sc_neurocore.chiplet.hierarchical_core import HierarchicalPartitioner
from sc_neurocore.chiplet.hierarchical_graph import (
    CSRGraph,
    CorrelationAwareGraph,
    CorrelationEdge,
    HierarchyLevel,
    LFSRSeedAllocator,
)
from sc_neurocore.chiplet.hierarchical_metrics import (
    _build_part_map as _build_part_map,
    calculate_boundary_scc,
    calculate_comm_volume,
    calculate_edge_cut,
    calculate_imbalance_ratio,
    calculate_mean_boundary_scc,
    calculate_total_boundary_scc,
)
from sc_neurocore.chiplet.hierarchical_reporting import (
    PartitionReport,
    build_partition_report,
)


_ensure_julia_kl_refine_loaded = _runtime._ensure_julia_kl_refine_loaded
_ensure_go_kl_refine_loaded = _runtime._ensure_go_kl_refine_loaded
_ensure_mojo_kl_refine_loaded = _runtime._ensure_mojo_kl_refine_loaded

__all__ = [
    "BoundarySyncConfig",
    "BoundarySyncProtocol",
    "CSRGraph",
    "CorrelationAwareGraph",
    "CorrelationEdge",
    "CorrelationLoadBalancer",
    "GhostCellManager",
    "HierarchicalPartitioner",
    "HierarchyLevel",
    "LFSRSeedAllocator",
    "LoadMetrics",
    "MigrationRecommendation",
    "PartitionReport",
    "RankMapper",
    "build_partition_report",
    "calculate_boundary_scc",
    "calculate_comm_volume",
    "calculate_edge_cut",
    "calculate_imbalance_ratio",
    "calculate_mean_boundary_scc",
    "calculate_total_boundary_scc",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__

del _public_name


def __getattr__(name: str) -> Any:
    """Expose historical read-only backend diagnostics."""
    runtime_names = {
        "_HAS_RUST_KL_REFINE",
        "_HAS_JULIA_KL_REFINE",
        "_HAS_GO_KL_REFINE",
        "_HAS_MOJO_KL_REFINE",
        "_rust_kl_refine",
        "_julia_kl_refine",
        "_go_kl_refine_lib",
        "_mojo_kl_refine_lib",
    }
    if name in runtime_names:
        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
