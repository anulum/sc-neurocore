# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partition reporting

"""Stable report model and composition for hierarchical partitioning."""

from __future__ import annotations

from dataclasses import dataclass

from sc_neurocore.chiplet.hierarchical_boundary import (
    BoundarySyncConfig,
    BoundarySyncProtocol,
)
from sc_neurocore.chiplet.hierarchical_graph import CorrelationAwareGraph
from sc_neurocore.chiplet.hierarchical_metrics import (
    calculate_boundary_scc,
    calculate_comm_volume,
    calculate_edge_cut,
    calculate_imbalance_ratio,
    calculate_mean_boundary_scc,
    calculate_total_boundary_scc,
)


@dataclass
class PartitionReport:
    """Metrics and deterministic seeds from one partitioning run."""

    num_partitions: int
    partition_sizes: list[int]
    edge_cut: int
    max_boundary_scc: float
    mean_boundary_scc: float
    total_boundary_scc: float
    imbalance_ratio: float
    comm_volume_bytes: int
    comm_messages: int
    seeds: list[int]
    scc_budget_violations: int = 0

    def summary(self) -> str:
        """Return a concise human-readable report."""
        return (
            f"Partitions: {self.num_partitions}, "
            f"Sizes: {self.partition_sizes}, "
            f"Edge cut: {self.edge_cut}, "
            f"Max boundary SCC: {self.max_boundary_scc:.4f}, "
            f"Mean boundary SCC: {self.mean_boundary_scc:.4f}, "
            f"Imbalance: {self.imbalance_ratio:.3f}, "
            f"Comm: {self.comm_volume_bytes} bytes / "
            f"{self.comm_messages} msgs"
        )


def build_partition_report(
    graph: CorrelationAwareGraph,
    partitions: list[list[int]],
    seeds: list[int],
    scc_budget: float = 0.1,
) -> PartitionReport:
    """Compose all partition metrics and SCC-budget violations."""
    communication = calculate_comm_volume(graph, partitions)
    sync = BoundarySyncProtocol(BoundarySyncConfig(max_boundary_scc_budget=scc_budget))
    violations = sync.check_scc_budget(graph, partitions)
    return PartitionReport(
        num_partitions=len(partitions),
        partition_sizes=[len(partition) for partition in partitions],
        edge_cut=calculate_edge_cut(graph, partitions),
        max_boundary_scc=calculate_boundary_scc(graph, partitions),
        mean_boundary_scc=calculate_mean_boundary_scc(graph, partitions),
        total_boundary_scc=calculate_total_boundary_scc(graph, partitions),
        imbalance_ratio=calculate_imbalance_ratio(partitions),
        comm_volume_bytes=communication["volume_bytes"],
        comm_messages=communication["messages"],
        seeds=seeds,
        scc_budget_violations=len(violations),
    )


__all__ = ["PartitionReport", "build_partition_report"]
