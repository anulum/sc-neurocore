# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet routing and link-analysis contracts

"""Routing-table, path, timing, congestion, and energy tests."""

from __future__ import annotations

import pytest

from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletTopology,
    CongestionReport,
    InterposerLink,
    InterposerTech,
    PackageEnergyReport,
    RoutingTable,
    adaptive_route,
    bandwidth_aware_route,
    compute_decorrelation_seeds,
    estimate_congestion,
    estimate_package_energy,
    find_disjoint_paths,
    link_energy_pj,
    make_torus,
    simulate_timing,
)


def test_routing_table_queries() -> None:
    table = RoutingTable(die_id=0)
    table.add_route(10, 1, 20)
    table.add_route(11, 2, 30)
    table.add_route(12, 1, 40)
    assert table.num_entries == 3
    assert len(table.routes_to_die(1)) == 2
    assert table.target_dies == [1, 2]


def test_decorrelation_seeds_are_unique_nonzero_and_bounded() -> None:
    seeds = compute_decorrelation_seeds(ChipletTopology.mesh_2d(4, 4))
    assert len(set(seeds.values())) == len(seeds)
    assert all(1 <= seed <= 65535 for seed in seeds.values())


class TestTiming:
    """Lowest-latency path aggregation."""

    def test_same_die(self) -> None:
        result = simulate_timing(ChipletTopology.ring(3), 0, 0)
        assert result is not None and result.total_latency_ns == 0.0

    def test_adjacent_and_multihop_paths(self) -> None:
        adjacent = simulate_timing(ChipletTopology.ring(4), 0, 1)
        multihop = simulate_timing(ChipletTopology.mesh_2d(2, 3), 0, 5)
        assert adjacent is not None and adjacent.path == [0, 1]
        assert multihop is not None and len(multihop.path) > 2

    def test_unreachable_returns_none(self) -> None:
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
        assert simulate_timing(topology, 0, 1) is None

    def test_path_accumulates_bandwidth_jitter_and_ber(self) -> None:
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1), ChipletDie(2)])
        topology.add_link(InterposerLink(0, 1, bandwidth_gbps=100.0, jitter_ns=0.2))
        topology.add_link(InterposerLink(1, 2, bandwidth_gbps=10.0, bit_error_rate=1e-9))
        result = simulate_timing(topology, 0, 2)
        assert result is not None
        assert result.min_bandwidth_gbps == 10.0
        assert result.max_jitter_ns == 0.2
        assert result.worst_ber == 1e-9


class TestEnergyAndCongestion:
    """Package communication estimates."""

    def test_energy_report_and_unit_conversion(self) -> None:
        topology = ChipletTopology.ring(4)
        report = estimate_package_energy(topology, bits_per_link=1000)
        assert isinstance(report, PackageEnergyReport)
        assert len(report.per_link_pj) == 4
        assert report.total_nj == report.total_pj / 1000.0

    def test_cowos_uses_less_energy_than_organic(self) -> None:
        cowos = link_energy_pj(InterposerLink.from_tech(0, 1, InterposerTech.COWOS), 256)
        organic = link_energy_pj(InterposerLink.from_tech(0, 1, InterposerTech.ORGANIC), 256)
        assert cowos < organic

    @pytest.mark.parametrize("bits", [-1])
    def test_negative_traffic_energy_fails(self, bits: int) -> None:
        with pytest.raises(ValueError, match="bits"):
            link_energy_pj(InterposerLink.from_tech(0, 1, InterposerTech.UCIE), bits)
        with pytest.raises(ValueError, match="bits_per_link"):
            estimate_package_energy(ChipletTopology.ring(2), bits)

    def test_congestion_identifies_narrow_link(self) -> None:
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
        topology.add_link(InterposerLink(0, 1, bandwidth_gbps=0.001))
        table = RoutingTable(die_id=0)
        for neuron_id in range(10):
            table.add_route(neuron_id, 1, neuron_id)
        report = estimate_congestion(topology, {0: table}, events_per_cycle=1000)
        assert isinstance(report, CongestionReport)
        assert report.bottleneck == (0, 1)
        assert report.max_utilisation > 1.0

    def test_zero_traffic_has_zero_utilisation(self) -> None:
        report = estimate_congestion(ChipletTopology.ring(3), {}, events_per_cycle=0)
        assert report.max_utilisation == 0.0

    def test_negative_event_rate_fails(self) -> None:
        with pytest.raises(ValueError, match="events_per_cycle"):
            estimate_congestion(ChipletTopology.ring(2), {}, -1)


class TestPathSelection:
    """Disjoint, congestion-aware, and bandwidth-aware route selection."""

    def test_disjoint_path_contracts(self) -> None:
        topology = make_torus(2, 2)
        assert find_disjoint_paths(topology, 0, 0) == [[0]]
        paths = find_disjoint_paths(topology, 0, 3, max_paths=2)
        assert paths and all(path[0] == 0 and path[-1] == 3 for path in paths)
        if len(paths) == 2:
            first = set(zip(paths[0], paths[0][1:]))
            second = set(zip(paths[1], paths[1][1:]))
            assert first.isdisjoint(second)

    def test_unreachable_and_zero_limit(self) -> None:
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
        assert find_disjoint_paths(topology, 0, 1) == []
        assert find_disjoint_paths(ChipletTopology.ring(2), 0, 1, max_paths=0) == []
        with pytest.raises(ValueError, match="max_paths"):
            find_disjoint_paths(ChipletTopology.ring(2), 0, 1, max_paths=-1)

    def test_adaptive_route_avoids_then_falls_back(self) -> None:
        topology = make_torus(2, 3)
        path = adaptive_route(
            topology,
            0,
            1,
            CongestionReport(utilisation={(0, 1): 0.95}),
            congestion_threshold=0.8,
        )
        assert path is not None and (0, 1) not in set(zip(path, path[1:]))
        fallback = adaptive_route(
            ChipletTopology.ring(2),
            0,
            1,
            CongestionReport(utilisation={(0, 1): 1.0}),
            congestion_threshold=0.0,
        )
        assert fallback == [0, 1]

    def test_bandwidth_route_success_failure_and_validation(self) -> None:
        topology = ChipletTopology.ring(3, InterposerTech.COWOS)
        assert bandwidth_aware_route(topology, 0, 0, 100.0) == [0]
        assert bandwidth_aware_route(topology, 0, 1, 50.0) == [0, 1]
        assert bandwidth_aware_route(topology, 0, 1, 500.0) is None
        with pytest.raises(ValueError, match="required_gbps"):
            bandwidth_aware_route(topology, 0, 1, -1.0)
        with pytest.raises(ValueError, match="congestion_threshold"):
            adaptive_route(topology, 0, 1, CongestionReport(), -1.0)
