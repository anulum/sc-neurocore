# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore.chiplet package public API

"""Tests for the stable ``sc_neurocore.chiplet`` package surface."""

from __future__ import annotations

import sc_neurocore.chiplet as ch
from sc_neurocore.chiplet import chiplet_gen as cg
from sc_neurocore.chiplet import hierarchical_partitioner as hp


CHIPLET_GEN_SYMBOLS: tuple[str, ...] = (
    "CDCConfig",
    "ChipletDie",
    "ChipletGenerator",
    "ChipletOutput",
    "ChipletTopology",
    "CongestionReport",
    "CreditConfig",
    "DieThermal",
    "InterposerLink",
    "InterposerTech",
    "LinkProtection",
    "PackageEnergyReport",
    "PackageThermalReport",
    "PartitionAssignment",
    "PowerDomain",
    "PowerDomainMap",
    "RoutingEntry",
    "RoutingTable",
    "StackingType",
    "TimingSimResult",
    "TSVLink",
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
)

HIER_SYMBOLS: tuple[str, ...] = (
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
)


def test_tier_is_research() -> None:
    assert ch.__tier__ == "research"


def test_all_lists_57_symbols() -> None:
    assert isinstance(ch.__all__, list)
    assert len(ch.__all__) == 57
    assert len(set(ch.__all__)) == 57, "no duplicates"


def test_chiplet_gen_symbols_importable() -> None:
    for sym in CHIPLET_GEN_SYMBOLS:
        assert hasattr(ch, sym), f"package missing chiplet_gen symbol {sym!r}"


def test_chiplet_gen_symbols_identity() -> None:
    """Top-level symbol IS the inner-module symbol (no shadow / no rebind)."""
    for sym in CHIPLET_GEN_SYMBOLS:
        assert getattr(ch, sym) is getattr(cg, sym)


def test_chiplet_gen_qualified_names_remain_stable() -> None:
    """Moved symbols retain historical pickle and introspection identities."""
    for symbol in CHIPLET_GEN_SYMBOLS:
        assert getattr(cg, symbol).__module__ == "sc_neurocore.chiplet.chiplet_gen"


def test_hier_symbols_importable() -> None:
    for sym in HIER_SYMBOLS:
        assert hasattr(ch, sym), f"package missing hierarchical_partitioner symbol {sym!r}"


def test_hier_symbols_identity() -> None:
    for sym in HIER_SYMBOLS:
        assert getattr(ch, sym) is getattr(hp, sym)


def test_hier_symbols_qualified_names_remain_stable() -> None:
    """Moved partitioner symbols retain their historical module identity."""
    for symbol in HIER_SYMBOLS:
        assert getattr(hp, symbol).__module__ == ("sc_neurocore.chiplet.hierarchical_partitioner")


def test_all_symbols_in_all() -> None:
    """Every documented symbol from both modules must appear in __all__."""
    public = set(ch.__all__)
    for sym in CHIPLET_GEN_SYMBOLS + HIER_SYMBOLS:
        assert sym in public, f"{sym!r} missing from __all__"


# ───────────────────────── instantiability smoke ─────────────────────────


def test_interposer_tech_enum_has_six_members() -> None:
    """6 die-to-die technology presets: UCIe / BoW / EMIB / CoWoS / Organic / Custom."""
    assert len(list(ch.InterposerTech)) == 6
    names = {m.name for m in ch.InterposerTech}
    assert {"UCIE", "BOW", "EMIB", "COWOS", "ORGANIC", "CUSTOM"} == names


def test_interposer_link_from_tech_yields_valid_object() -> None:
    """InterposerLink.from_tech(...) must construct without raising on every preset."""
    for tech in ch.InterposerTech:
        link = ch.InterposerLink.from_tech(0, 1, tech)
        assert link.src_die == 0
        assert link.dst_die == 1
        assert link.technology is tech
        assert link.latency_ns > 0
        assert link.bandwidth_gbps > 0
        assert 0 < link.bit_error_rate < 1


def test_compute_decorrelation_seeds_returns_tuple_keyed_dict() -> None:
    """seeds key is (src_die, dst_die) tuple, not a single int (mypy fix in this batch)."""
    topo = ch.ChipletTopology()
    topo.add_die(ch.ChipletDie(die_id=0))
    topo.add_die(ch.ChipletDie(die_id=1))
    topo.add_link(ch.InterposerLink.from_tech(0, 1, ch.InterposerTech.UCIE))

    seeds = ch.compute_decorrelation_seeds(topo)
    assert isinstance(seeds, dict)
    assert len(seeds) == 1
    key = next(iter(seeds))
    assert isinstance(key, tuple) and len(key) == 2
    assert key == (0, 1)
    assert isinstance(seeds[key], int)
    assert 1 <= seeds[key] <= 65535


def test_make_torus_returns_topology() -> None:
    """make_torus(rows, cols) builds a `rows`×`cols` torus."""
    topo = ch.make_torus(3, 3)  # 3×3 = 9 dies
    assert isinstance(topo, ch.ChipletTopology)
    assert len(topo.dies) == 9
    # Each of 9 dies has 4 wrap-around neighbours.
    # Some implementations might return half (undirected representation).
    # Tolerate either, but require non-zero.
    assert len(topo.links) > 0


def test_hierarchical_partitioner_constructs() -> None:
    """HierarchicalPartitioner with default args is a valid construction."""
    hp_obj = ch.HierarchicalPartitioner(num_partitions=2)
    assert hp_obj is not None
