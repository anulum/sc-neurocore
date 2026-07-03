# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Defensive-branch coverage for chiplet_gen edge paths

"""Edge-case tests targeting the 12 lines pytest --cov flagged
uncovered in `chiplet_gen.py` after the multi-lang KL refine work
landed. Every test has a single, well-defined target line; comments
on each test class identify the line range it pins.

The defensive guards covered here:
  - LFSR seed clamps (`seed = 1` when `(0xACE1 + die_id*7919) & 0xFFFF == 0`)
  - ClockDomainCrossing.ratio with zero divisor
  - compute_cdc_configs / _build_conductance_matrix `continue` paths
    when topology references missing dies
  - simulate_thermal `die_state[die_id]` override branch
  - find_route_with_congestion fallback to no-exclusion BFS
  - find_route_min_bandwidth visited-skip + queue-append paths
  - ChipletGenerator.compile cross-die-skip path
"""

from __future__ import annotations

import pytest

from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletTopology,
    InterposerLink,
    InterposerTech,
    make_torus,
    simulate_thermal,
)
from sc_neurocore.chiplet.chiplet_gen import (
    CDCConfig,
    CongestionReport,
    DieThermal,
    PartitionAssignment,
    PowerDomain,
    adaptive_route,
    bandwidth_aware_route,
    compute_cdc_configs,
)


# ────────────────────────────────────────────────────────────────────
# LFSR seed-clamp branches (lines 156, 175, 188, 793)
# ────────────────────────────────────────────────────────────────────

# Computed via: pow(7919, -1, 65536) * (-0xACE1 % 65536) % 65536 → 3793
# Any topology that includes die_id=3793 will hit `seed == 0 → seed = 1`.
SEED_ZERO_DIE_ID = 3793


class TestLfsrSeedClamp:
    """The four `if seed == 0: seed = 1` guards in mesh_2d / ring /
    star / make_torus topology factories trigger when die_id=3793
    (and 3793 mod 65536 multiples). Pin each path."""

    def test_mesh_2d_hits_seed_zero(self) -> None:
        # 60×64 mesh has die ids 0..3839, including 3793.
        topo = ChipletTopology.mesh_2d(rows=60, cols=64)
        target = next(d for d in topo.dies if d.die_id == SEED_ZERO_DIE_ID)
        assert target.lfsr_seed == 1

    def test_ring_hits_seed_zero(self) -> None:
        topo = ChipletTopology.ring(n_dies=SEED_ZERO_DIE_ID + 1)
        assert topo.dies[SEED_ZERO_DIE_ID].lfsr_seed == 1

    def test_star_hits_seed_zero(self) -> None:
        topo = ChipletTopology.star(n_dies=SEED_ZERO_DIE_ID + 1)
        assert topo.dies[SEED_ZERO_DIE_ID].lfsr_seed == 1

    def test_make_torus_hits_seed_zero(self) -> None:
        # 64x60 torus includes die_id=3793.
        topo = make_torus(rows=64, cols=60)
        target = next(d for d in topo.dies if d.die_id == SEED_ZERO_DIE_ID)
        assert target.lfsr_seed == 1


# ────────────────────────────────────────────────────────────────────
# CDC ratio zero-divisor guard (line 823)
# ────────────────────────────────────────────────────────────────────


class TestClockDomainCrossingZero:
    def test_ratio_returns_one_when_dst_clock_zero(self) -> None:
        cdc = CDCConfig(src_clk_mhz=200.0, dst_clk_mhz=0.0)
        assert cdc.ratio == 1.0

    def test_ratio_uses_real_division_when_dst_nonzero(self) -> None:
        cdc = CDCConfig(src_clk_mhz=200.0, dst_clk_mhz=100.0)
        assert cdc.ratio == 2.0


# ────────────────────────────────────────────────────────────────────
# compute_cdc_configs missing-die guard (line 838)
# ────────────────────────────────────────────────────────────────────


class TestCdcConfigsMissingDie:
    def test_continues_when_link_references_unknown_die(self) -> None:
        # Topology has dies 0, 1; link references die 99 (missing).
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0, clock_mhz=100.0))
        topo.add_die(ChipletDie(die_id=1, clock_mhz=100.0))
        topo.add_link(InterposerLink.from_tech(0, 99, InterposerTech.UCIE))
        topo.add_link(InterposerLink.from_tech(0, 1, InterposerTech.UCIE))
        cfgs = compute_cdc_configs(topo)
        # Only the (0, 1) link should produce a CDCConfig.
        assert (0, 1) in cfgs
        assert (0, 99) not in cfgs


# ────────────────────────────────────────────────────────────────────
# _build_conductance_matrix missing-die guard (line 967)
# ────────────────────────────────────────────────────────────────────


class TestThermalConductanceMissingDie:
    def test_simulate_thermal_skips_link_with_missing_die(self) -> None:
        # Dies 0 and 1; link references die 99.
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0))
        topo.add_die(ChipletDie(die_id=1))
        topo.add_link(InterposerLink.from_tech(0, 99, InterposerTech.UCIE))
        # Should run without raising despite the dangling link.
        report = simulate_thermal(topo, power_per_die_mw={0: 100.0, 1: 100.0})
        assert report is not None


# ────────────────────────────────────────────────────────────────────
# simulate_thermal die_state override branch (line 1085)
# ────────────────────────────────────────────────────────────────────


class TestSimulateThermalCustomDieState:
    def test_uses_provided_die_state_dict(self) -> None:
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0))
        topo.add_die(ChipletDie(die_id=1))
        # Override only die 0; die 1 falls through to default branch.
        custom = {0: DieThermal(die_id=0, r_to_ambient_k_per_w=10.0)}
        report = simulate_thermal(
            topo,
            power_per_die_mw={0: 1000.0, 1: 100.0},
            die_state=custom,
        )
        assert report is not None
        # Die 0's higher R_thermal → higher temperature than die 1.
        t0 = report.die_temps[0]
        t1 = report.die_temps[1]
        assert t0 > t1


# ────────────────────────────────────────────────────────────────────
# find_route_with_congestion fallback to no-exclusion (line 1155)
# ────────────────────────────────────────────────────────────────────


class TestAdaptiveRouteCongestionFallback:
    def test_falls_back_when_all_routes_congested(self) -> None:
        # 3-die ring 0–1–2–0, every link saturated → primary BFS fails,
        # fallback (line 1155) ignores congestion and finds a path.
        topo = ChipletTopology()
        for i in range(3):
            topo.add_die(ChipletDie(die_id=i))
        for s, d in [(0, 1), (1, 2), (2, 0)]:
            topo.add_link(InterposerLink.from_tech(s, d, InterposerTech.UCIE))
        congestion = CongestionReport()
        for link in topo.links:
            congestion.utilisation[(link.src_die, link.dst_die)] = 1.0
        # Threshold 0.5 → primary excludes every link, fallback wins.
        path = adaptive_route(
            topo,
            src_die=0,
            dst_die=2,
            congestion=congestion,
            congestion_threshold=0.5,
        )
        assert path is not None
        assert path[0] == 0 and path[-1] == 2


# ────────────────────────────────────────────────────────────────────
# bandwidth_aware_route visited + queue-append (lines 1247, 1253)
# ────────────────────────────────────────────────────────────────────


class TestBandwidthAwareRoute:
    def test_visited_skip_and_queue_extension(self) -> None:
        # 4-die mesh 0↔1↔2↔3 + 1↔3 short-cut, all links 50 Gbps.
        topo = ChipletTopology()
        for i in range(4):
            topo.add_die(ChipletDie(die_id=i))
        for s, d in [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)]:
            link = InterposerLink.from_tech(s, d, InterposerTech.UCIE)
            link.bandwidth_gbps = 50.0
            topo.add_link(link)
        # required 30 Gbps ≤ every link → BFS extends queue past die 1
        # (queue.append, line 1253). Two paths reach die 3 → visited-skip
        # at line 1247.
        path = bandwidth_aware_route(
            topo,
            src_die=0,
            dst_die=3,
            required_gbps=30.0,
        )
        assert path is not None
        assert path[0] == 0 and path[-1] == 3

    def test_returns_none_when_bandwidth_insufficient(self) -> None:
        topo = ChipletTopology()
        for i in range(2):
            topo.add_die(ChipletDie(die_id=i))
        link = InterposerLink.from_tech(0, 1, InterposerTech.UCIE)
        link.bandwidth_gbps = 10.0
        topo.add_link(link)
        # required 100 Gbps > 10 Gbps available → no path.
        path = bandwidth_aware_route(
            topo,
            src_die=0,
            dst_die=1,
            required_gbps=100.0,
        )
        assert path is None


# ────────────────────────────────────────────────────────────────────
# PartitionAssignment.to_routing_tables cross-die skip (line 1485)
# ────────────────────────────────────────────────────────────────────


class TestPartitionAssignmentCrossDieSkip:
    def test_connectivity_with_unmapped_neuron_skipped(self) -> None:
        # Dies 0+1, neurons 0/1 → die 0; neurons 2/3 → die 1.
        pa = PartitionAssignment(die_assignments={0: [0, 1], 1: [2, 3]})
        connectivity: list[tuple[int, int, int]] = [
            (0, 2, 256),  # cross-die: die 0 → die 1 ✓
            (99, 3, 256),  # 99 unmapped → continue (line 1485)
            (0, 1, 256),  # same-die: die 0 → die 0 → continue (1487)
        ]
        tables = pa.to_routing_tables(connectivity)
        # Only the cross-die route appears.
        assert 0 in tables
        assert len(tables[0].entries) == 1
        assert tables[0].entries[0].dst_die == 1
        assert tables[0].entries[0].src_neuron == 0
        assert tables[0].entries[0].dst_neuron == 2


class TestPowerDomainValidation:
    def test_duplicate_die_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            PowerDomain(domain_id=0, die_ids=[1, 1], voltage_mv=800)
