# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.chiplet package public API surface

"""Multi-die chiplet generation and hierarchical graph partitioning.

Tier: research.

The historical ``chiplet_gen`` module remains an import-compatible facade over
focused responsibilities:

- ``topology`` — die, interposer, planar-topology, and stack models.
- ``routing`` — routes, timing, energy, and congestion analysis.
- ``thermal`` — steady-state and transient package thermal solving.
- ``rtl`` — connected package SystemVerilog and XDC generation.
- ``link_protocols`` — CDC, CRC, and credit-flow contracts.
- ``power`` — voltage-island ownership and power-gating RTL.
- ``partition`` — neuron assignment to AER routing translation.
- ``hierarchical_partitioner`` — exascale graph partitioner (CSR
  graph, correlation-aware partitioning, LFSR seed allocator,
  ghost-cell manager, boundary synchronisation, migration
  recommendations, partition-quality metrics).
"""

from sc_neurocore.chiplet.chiplet_gen import (
    CDCConfig,
    ChipletDie,
    ChipletGenerator,
    ChipletOutput,
    ChipletTopology,
    CongestionReport,
    CreditConfig,
    DieThermal,
    InterposerLink,
    InterposerTech,
    LinkProtection,
    PackageEnergyReport,
    PackageThermalReport,
    PartitionAssignment,
    PowerDomain,
    PowerDomainMap,
    RoutingEntry,
    RoutingTable,
    StackingType,
    TimingSimResult,
    TSVLink,
    adaptive_route,
    add_3d_stack,
    bandwidth_aware_route,
    compute_cdc_configs,
    compute_decorrelation_seeds,
    emit_credit_controller_sv,
    emit_crc32_sv,
    emit_power_gating_sv,
    estimate_congestion,
    estimate_package_energy,
    find_disjoint_paths,
    link_energy_pj,
    make_torus,
    simulate_thermal,
    simulate_timing,
)
from sc_neurocore.chiplet.hierarchical_partitioner import (
    BoundarySyncConfig,
    BoundarySyncProtocol,
    CSRGraph,
    CorrelationAwareGraph,
    CorrelationEdge,
    CorrelationLoadBalancer,
    GhostCellManager,
    HierarchicalPartitioner,
    HierarchyLevel,
    LFSRSeedAllocator,
    LoadMetrics,
    MigrationRecommendation,
    PartitionReport,
    RankMapper,
    build_partition_report,
    calculate_boundary_scc,
    calculate_comm_volume,
    calculate_edge_cut,
    calculate_imbalance_ratio,
    calculate_mean_boundary_scc,
    calculate_total_boundary_scc,
)

__tier__ = "research"

__all__ = [
    # ─── chiplet compatibility surface — enums
    "InterposerTech",
    "PowerDomain",
    "StackingType",
    # ─── chiplet compatibility surface — dataclasses
    "CDCConfig",
    "ChipletDie",
    "ChipletOutput",
    "ChipletTopology",
    "CongestionReport",
    "CreditConfig",
    "DieThermal",
    "InterposerLink",
    "LinkProtection",
    "PackageEnergyReport",
    "PackageThermalReport",
    "PartitionAssignment",
    "PowerDomainMap",
    "RoutingEntry",
    "RoutingTable",
    "TimingSimResult",
    "TSVLink",
    # ─── chiplet compatibility surface — generation + analysis
    "ChipletGenerator",
    "adaptive_route",
    "add_3d_stack",
    "bandwidth_aware_route",
    "compute_cdc_configs",
    "compute_decorrelation_seeds",
    "emit_credit_controller_sv",
    "emit_crc32_sv",
    "emit_power_gating_sv",
    "estimate_congestion",
    "estimate_package_energy",
    "find_disjoint_paths",
    "link_energy_pj",
    "make_torus",
    "simulate_thermal",
    "simulate_timing",
    # ─── hierarchical_partitioner — enums
    "HierarchyLevel",
    # ─── hierarchical_partitioner — graphs / dataclasses
    "BoundarySyncConfig",
    "CSRGraph",
    "CorrelationAwareGraph",
    "CorrelationEdge",
    "LoadMetrics",
    "MigrationRecommendation",
    "PartitionReport",
    # ─── hierarchical_partitioner — orchestrators
    "BoundarySyncProtocol",
    "CorrelationLoadBalancer",
    "GhostCellManager",
    "HierarchicalPartitioner",
    "LFSRSeedAllocator",
    "RankMapper",
    "build_partition_report",
    "calculate_boundary_scc",
    "calculate_comm_volume",
    "calculate_edge_cut",
    "calculate_imbalance_ratio",
    "calculate_mean_boundary_scc",
    "calculate_total_boundary_scc",
]
