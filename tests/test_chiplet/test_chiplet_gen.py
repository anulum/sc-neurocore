# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet/Interposer Generator Tests

import sys
import os

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "sc_neurocore", "chiplet")
)

from chiplet_gen import (
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
    RoutingTable,
    StackingType,
    TSVLink,
    adaptive_route,
    add_3d_stack,
    bandwidth_aware_route,
    compute_cdc_configs,
    compute_decorrelation_seeds,
    emit_crc32_sv,
    emit_credit_controller_sv,
    emit_power_gating_sv,
    estimate_congestion,
    estimate_package_energy,
    find_disjoint_paths,
    link_energy_pj,
    make_torus,
    simulate_thermal,
    simulate_timing,
)


# ── InterposerLink Tests ────────────────────────────────────────────


class TestInterposerLink:
    def test_from_tech_ucie(self):
        link = InterposerLink.from_tech(0, 1, InterposerTech.UCIE)
        assert link.latency_ns == 2.0
        assert link.bandwidth_gbps == 32.0

    def test_from_tech_cowos(self):
        link = InterposerLink.from_tech(0, 1, InterposerTech.COWOS)
        assert link.latency_ns < 1.0
        assert link.bandwidth_gbps > 100

    def test_from_tech_organic(self):
        link = InterposerLink.from_tech(0, 1, InterposerTech.ORGANIC)
        assert link.latency_ns > 3.0

    def test_latency_cycles(self):
        link = InterposerLink(src_die=0, dst_die=1, latency_ns=10.0)
        assert link.latency_cycles == 2  # 10ns / 5ns per cycle

    def test_fifo_depth(self):
        link = InterposerLink(src_die=0, dst_die=1, jitter_ns=5.0)
        assert link.fifo_depth_log2 >= 3

    def test_all_technologies(self):
        for tech in InterposerTech:
            link = InterposerLink.from_tech(0, 1, tech)
            assert link.latency_ns > 0
            assert link.bandwidth_gbps > 0
            assert link.bit_error_rate > 0


# ── ChipletDie Tests ─────────────────────────────────────────────────


class TestChipletDie:
    def test_defaults(self):
        die = ChipletDie(die_id=0)
        assert die.clock_mhz == 200.0
        assert die.n_neurons == 128

    def test_clock_period(self):
        die = ChipletDie(die_id=0, clock_mhz=100.0)
        assert die.clock_period_ns == 10.0

    def test_custom_seed(self):
        die = ChipletDie(die_id=5, lfsr_seed=0xBEEF)
        assert die.lfsr_seed == 0xBEEF


# ── ChipletTopology Tests ───────────────────────────────────────────


class TestChipletTopology:
    def test_mesh_2d(self):
        topo = ChipletTopology.mesh_2d(2, 3, InterposerTech.UCIE)
        assert topo.num_dies == 6
        assert len(topo.links) == 7  # 2*3 - col+row edges: 3+2+2 = 7

    def test_ring(self):
        topo = ChipletTopology.ring(4, InterposerTech.EMIB)
        assert topo.num_dies == 4
        assert len(topo.links) == 4

    def test_get_links_from(self):
        topo = ChipletTopology.ring(3)
        links = topo.get_links_from(0)
        assert len(links) == 1
        assert links[0].dst_die == 1

    def test_get_links_to(self):
        topo = ChipletTopology.ring(3)
        links = topo.get_links_to(0)
        assert len(links) == 1  # From die 2 → die 0

    def test_get_die(self):
        topo = ChipletTopology.mesh_2d(2, 2)
        die = topo.get_die(3)
        assert die is not None
        assert die.die_id == 3

    def test_unique_seeds(self):
        topo = ChipletTopology.mesh_2d(3, 3)
        seeds = [d.lfsr_seed for d in topo.dies]
        assert len(set(seeds)) == len(seeds), "LFSR seeds must be unique"


# ── RoutingTable Tests ───────────────────────────────────────────────


class TestRoutingTable:
    def test_add_route(self):
        rt = RoutingTable(die_id=0)
        rt.add_route(10, 1, 20)
        assert rt.num_entries == 1

    def test_routes_to_die(self):
        rt = RoutingTable(die_id=0)
        rt.add_route(10, 1, 20)
        rt.add_route(11, 2, 30)
        rt.add_route(12, 1, 40)
        assert len(rt.routes_to_die(1)) == 2

    def test_target_dies(self):
        rt = RoutingTable(die_id=0)
        rt.add_route(0, 1, 0)
        rt.add_route(1, 3, 10)
        assert rt.target_dies == [1, 3]


# ── Decorrelation Seeds Tests ────────────────────────────────────────


class TestDecorrelation:
    def test_unique_seeds(self):
        topo = ChipletTopology.mesh_2d(3, 3)
        seeds = compute_decorrelation_seeds(topo)
        values = list(seeds.values())
        assert len(set(values)) == len(values), "Seeds must be unique"

    def test_nonzero_seeds(self):
        topo = ChipletTopology.ring(8)
        seeds = compute_decorrelation_seeds(topo)
        for s in seeds.values():
            assert s > 0

    def test_seed_range(self):
        topo = ChipletTopology.mesh_2d(4, 4)
        seeds = compute_decorrelation_seeds(topo)
        for s in seeds.values():
            assert 1 <= s <= 65535


# ── SystemVerilog Generator Tests ────────────────────────────────────


class TestChipletGenerator:
    def test_generate_mesh(self):
        topo = ChipletTopology.mesh_2d(2, 2)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        assert isinstance(out, ChipletOutput)

    def test_top_has_module(self):
        topo = ChipletTopology.ring(3)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        assert "module sc_chiplet_top" in out.top_sv

    def test_top_has_die_count(self):
        topo = ChipletTopology.mesh_2d(2, 3)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        assert "Dies: 6" in out.top_sv

    def test_die_module_has_lfsr(self):
        topo = ChipletTopology.ring(2)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        assert "lfsr" in out.die_modules[0]

    def test_die_module_has_aer_router(self):
        topo = ChipletTopology.ring(2)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        assert "sc_aer_router" in out.die_modules[0]

    def test_bridge_has_async_fifo(self):
        topo = ChipletTopology.ring(2)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        bridge = out.link_bridges[(0, 1)]
        assert "sc_async_fifo" in bridge

    def test_bridge_has_decorrelation(self):
        topo = ChipletTopology.ring(2)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        bridge = out.link_bridges[(0, 1)]
        assert "decorrelated" in bridge
        assert "lfsr" in bridge

    def test_bridge_has_latency_model(self):
        topo = ChipletTopology.ring(2, InterposerTech.COWOS)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        bridge = out.link_bridges[(0, 1)]
        assert "delay_pipe" in bridge
        assert "LATENCY_CYC" in bridge

    def test_bridge_has_technology_comment(self):
        topo = ChipletTopology.ring(2, InterposerTech.EMIB)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        bridge = out.link_bridges[(0, 1)]
        assert "EMIB" in bridge

    def test_routing_table_sv(self):
        topo = ChipletTopology.ring(2)
        rt = RoutingTable(die_id=0)
        rt.add_route(5, 1, 10, 256)
        gen = ChipletGenerator()
        out = gen.generate(topo, routing={0: rt})
        assert 0 in out.routing_tables
        assert "rt_target_die" in out.routing_tables[0]

    def test_constraints_xdc(self):
        topo = ChipletTopology.mesh_2d(2, 2)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        assert "create_clock" in out.constraints_xdc
        assert "set_max_delay" in out.constraints_xdc

    def test_filelist(self):
        topo = ChipletTopology.ring(3)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        assert "sc_chiplet_top.sv" in out.filelist
        assert len(out.filelist) > 5

    def test_to_dict(self):
        topo = ChipletTopology.ring(2)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        d = out.to_dict()
        assert "sc_chiplet_top.sv" in d
        assert "chiplet_constraints.xdc" in d

    def test_spdx_in_sv(self):
        topo = ChipletTopology.ring(2)
        gen = ChipletGenerator()
        out = gen.generate(topo)
        for name, content in out.to_dict().items():
            if name.endswith(".sv"):
                assert "SPDX" in content, f"Missing SPDX in {name}"


# ── Timing Simulator Tests ──────────────────────────────────────────


class TestTimingSimulator:
    def test_same_die(self):
        topo = ChipletTopology.ring(3)
        result = simulate_timing(topo, 0, 0)
        assert result is not None
        assert result.total_latency_ns == 0.0

    def test_adjacent_dies(self):
        topo = ChipletTopology.ring(4, InterposerTech.UCIE)
        result = simulate_timing(topo, 0, 1)
        assert result is not None
        assert result.total_latency_ns == 2.0
        assert result.path == [0, 1]

    def test_multi_hop(self):
        topo = ChipletTopology.mesh_2d(2, 3, InterposerTech.UCIE)
        result = simulate_timing(topo, 0, 5)
        assert result is not None
        assert result.total_latency_ns > 2.0
        assert len(result.path) > 2

    def test_unreachable(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(0))
        topo.add_die(ChipletDie(1))
        result = simulate_timing(topo, 0, 1)
        assert result is None

    def test_bandwidth_bottleneck(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(0))
        topo.add_die(ChipletDie(1))
        topo.add_die(ChipletDie(2))
        topo.add_link(InterposerLink(0, 1, bandwidth_gbps=100.0))
        topo.add_link(InterposerLink(1, 2, bandwidth_gbps=10.0))
        result = simulate_timing(topo, 0, 2)
        assert result is not None
        assert result.min_bandwidth_gbps == 10.0


# ── Star Topology Tests ─────────────────────────────────────────────


class TestStarTopology:
    def test_star_die_count(self):
        topo = ChipletTopology.star(5)
        assert topo.num_dies == 5

    def test_star_link_count(self):
        topo = ChipletTopology.star(5)
        # 4 spokes × 2 directions = 8 links
        assert len(topo.links) == 8

    def test_star_hub_connectivity(self):
        topo = ChipletTopology.star(4)
        from_hub = topo.get_links_from(0)
        to_hub = topo.get_links_to(0)
        assert len(from_hub) == 3
        assert len(to_hub) == 3

    def test_star_unique_seeds(self):
        topo = ChipletTopology.star(6)
        seeds = [d.lfsr_seed for d in topo.dies]
        assert len(set(seeds)) == len(seeds)

    def test_star_timing_through_hub(self):
        topo = ChipletTopology.star(4, InterposerTech.UCIE)
        result = simulate_timing(topo, 1, 2)
        assert result is not None
        assert len(result.path) == 3  # 1 → 0 → 2


# ── Link Energy Model Tests ─────────────────────────────────────────


class TestLinkEnergy:
    def test_energy_per_link(self):
        link = InterposerLink.from_tech(0, 1, InterposerTech.COWOS)
        e = link_energy_pj(link, 256)
        assert e > 0

    def test_cowos_cheapest(self):
        e_cowos = link_energy_pj(InterposerLink.from_tech(0, 1, InterposerTech.COWOS), 256)
        e_organic = link_energy_pj(InterposerLink.from_tech(0, 1, InterposerTech.ORGANIC), 256)
        assert e_cowos < e_organic

    def test_package_energy(self):
        topo = ChipletTopology.ring(4, InterposerTech.UCIE)
        report = estimate_package_energy(topo, bits_per_link=256)
        assert isinstance(report, PackageEnergyReport)
        assert report.total_pj > 0
        assert len(report.per_link_pj) == 4

    def test_total_nj_conversion(self):
        topo = ChipletTopology.ring(2)
        report = estimate_package_energy(topo, bits_per_link=1000)
        assert abs(report.total_nj - report.total_pj / 1000.0) < 1e-10


# ── Congestion Estimator Tests ──────────────────────────────────────


class TestCongestion:
    def test_no_traffic(self):
        topo = ChipletTopology.ring(3)
        report = estimate_congestion(topo, {}, events_per_cycle=0)
        assert report.max_utilisation == 0.0

    def test_with_traffic(self):
        topo = ChipletTopology.ring(3)
        rt0 = RoutingTable(die_id=0)
        rt0.add_route(0, 1, 10)
        report = estimate_congestion(topo, {0: rt0}, events_per_cycle=100)
        assert isinstance(report, CongestionReport)
        assert report.max_utilisation > 0

    def test_bottleneck_identified(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(0))
        topo.add_die(ChipletDie(1))
        topo.add_link(InterposerLink(0, 1, bandwidth_gbps=0.001))
        rt = RoutingTable(die_id=0)
        for i in range(10):
            rt.add_route(i, 1, i)
        report = estimate_congestion(topo, {0: rt}, events_per_cycle=1000)
        assert report.bottleneck == (0, 1)
        assert report.max_utilisation > 1.0


# ── Fault-Tolerant Routing Tests ────────────────────────────────────


class TestDisjointPaths:
    def test_same_die(self):
        topo = ChipletTopology.ring(3)
        paths = find_disjoint_paths(topo, 0, 0)
        assert paths == [[0]]

    def test_ring_two_paths(self):
        # Ring has two directions
        topo = ChipletTopology.ring(4)
        paths = find_disjoint_paths(topo, 0, 2, max_paths=2)
        assert len(paths) >= 1
        assert paths[0][0] == 0 and paths[0][-1] == 2

    def test_mesh_multiple_paths(self):
        topo = ChipletTopology.mesh_2d(2, 3)  # 0-1-2 / 3-4-5
        paths = find_disjoint_paths(topo, 0, 5, max_paths=2)
        assert len(paths) >= 1

    def test_unreachable(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(0))
        topo.add_die(ChipletDie(1))
        paths = find_disjoint_paths(topo, 0, 1)
        assert paths == []

    def test_disjoint_links(self):
        topo = ChipletTopology.ring(4)
        paths = find_disjoint_paths(topo, 0, 2, max_paths=2)
        if len(paths) == 2:
            edges_0 = set(zip(paths[0][:-1], paths[0][1:]))
            edges_1 = set(zip(paths[1][:-1], paths[1][1:]))
            assert edges_0.isdisjoint(edges_1)


# ── Torus Topology Tests ─────────────────────────────────────────────


class TestTorusTopology:
    def test_torus_die_count(self):
        topo = make_torus(3, 3)
        assert topo.num_dies == 9

    def test_torus_link_count(self):
        topo = make_torus(2, 3)
        # Each die has 2 outgoing links (right + down), all wrap
        assert len(topo.links) == 2 * 3 * 2

    def test_torus_wraparound(self):
        topo = make_torus(2, 3)
        # Die 2 (row 0, col 2) should link to die 0 (row 0, col 0)
        links = topo.get_links_from(2)
        dst_dies = [l.dst_die for l in links]
        assert 0 in dst_dies  # wrap-around right

    def test_torus_unique_seeds(self):
        topo = make_torus(3, 3)
        seeds = [d.lfsr_seed for d in topo.dies]
        assert len(set(seeds)) == len(seeds)

    def test_torus_fully_connected(self):
        topo = make_torus(3, 3)
        result = simulate_timing(topo, 0, 8)
        assert result is not None


# ── CDC Config Tests ─────────────────────────────────────────────────


class TestCDCConfig:
    def test_same_clock(self):
        topo = ChipletTopology.ring(3)
        configs = compute_cdc_configs(topo)
        for cfg in configs.values():
            assert cfg.is_mesochronous
            assert cfg.sync_stages == 2

    def test_different_clocks(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0, clock_mhz=200.0))
        topo.add_die(ChipletDie(die_id=1, clock_mhz=100.0))
        topo.add_link(InterposerLink.from_tech(0, 1, InterposerTech.UCIE))
        configs = compute_cdc_configs(topo)
        cfg = configs[(0, 1)]
        assert not cfg.is_mesochronous
        assert cfg.sync_stages == 3
        assert cfg.ratio == 2.0


# ── Thermal Model Tests ──────────────────────────────────────────────


class TestThermalModel:
    """Conductance-matrix thermal solver tests.

    The previous DieThermal.step() single-equation API was replaced
    2026-04-17 with a HotSpot-style package-level solver
    (`feedback_sophisticated_from_start.md`). These tests exercise
    the new solver's physics invariants.
    """

    def test_single_die_no_neighbours_obeys_ohm_law(self):
        """1-die package with no links: T = T_amb + P · R_amb.

        With the default DieThermal r_to_ambient_k_per_w = 1.5 and
        100 mW power, expected T = 25 + 0.1 W · 1.5 K/W = 25.15 °C.
        """
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0))
        report = simulate_thermal(topo, power_per_die_mw={0: 100.0}, ambient_c=25.0)
        assert abs(report.die_temps[0] - 25.15) < 1e-6

    def test_two_die_with_link_couples_temperatures(self):
        """Two coupled dies: hot-die's heat flows to cold-die through the link.

        Without coupling, die 0 (10 W) would be hot and die 1 (0 W)
        would be at ambient. WITH coupling, die 1 is heated by the
        bond, and die 0 is cooled. The conductance-matrix solver
        captures this — a single-equation model would not.
        """
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0))
        topo.add_die(ChipletDie(die_id=1))
        topo.add_link(InterposerLink.from_tech(0, 1, InterposerTech.UCIE))
        report = simulate_thermal(
            topo,
            power_per_die_mw={0: 10_000.0, 1: 0.0},  # 10 W vs 0 W
            ambient_c=25.0,
        )
        t0, t1 = report.die_temps[0], report.die_temps[1]
        # Die 1 (zero power) MUST be heated above ambient by conduction.
        assert t1 > 25.0 + 1e-3, f"die 1 not heated by neighbour: T1={t1}"
        # Die 0 MUST be cooler than the no-coupling case (T0_solo).
        # Solo: T0_solo = 25 + 10 W · 1.5 K/W = 40 °C
        # Coupled: should be < 40.
        assert t0 < 40.0, f"die 0 not cooled by coupling: T0={t0}"
        # Energy conservation: total heat dissipated == ambient flux out.
        # Σ (T_i - T_amb) / R_amb,i  ==  Σ P_i  (steady state, K · W/K = W)
        # With identical R_amb = 1.5 K/W:
        outflow_w = ((t0 - 25.0) + (t1 - 25.0)) / 1.5
        assert abs(outflow_w - 10.0) < 1e-6, f"power balance broken: {outflow_w} W out vs 10 W in"

    def test_throttled_flag_set_when_steady_state_above_max(self):
        """High power → die exceeds max_temperature_c → throttled."""
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0))
        report = simulate_thermal(
            topo,
            power_per_die_mw={0: 100_000.0},  # 100 W → way above limit
            ambient_c=25.0,
        )
        assert 0 in report.throttled_dies
        assert report.die_temps[0] > 100.0

    def test_not_throttled_at_low_power(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0))
        report = simulate_thermal(topo, power_per_die_mw={0: 100.0}, ambient_c=25.0)
        assert 0 not in report.throttled_dies

    def test_conductance_matrix_is_symmetric(self):
        """Off-diagonal G must be symmetric — heat flow direction-independent."""
        topo = ChipletTopology.ring(4)
        report = simulate_thermal(topo)
        G = report.conductance_matrix
        assert G is not None
        np.testing.assert_allclose(G, G.T, atol=1e-12)

    def test_conductance_matrix_zero_diagonal(self):
        """Off-diagonal storage MUST have zero on the diagonal — the
        diagonal effective conductance lives in the solver's local
        `diag` variable, not in `G_off`.
        """
        topo = ChipletTopology.ring(4)
        report = simulate_thermal(topo)
        G = report.conductance_matrix
        np.testing.assert_array_equal(np.diag(G), np.zeros(4))

    def test_higher_R_link_couples_less_strongly(self):
        """ORGANIC bond (8 K/W) should couple less than CoWoS (0.3 K/W).

        With the same power profile, the cold die heats up MORE on
        the low-R bond (better heat spreading from hot neighbour).
        """
        # Hot die @ 10 W, cold die @ 0 W
        powers = {0: 10_000.0, 1: 0.0}

        topo_lo = ChipletTopology()
        topo_lo.add_die(ChipletDie(die_id=0))
        topo_lo.add_die(ChipletDie(die_id=1))
        topo_lo.add_link(InterposerLink.from_tech(0, 1, InterposerTech.COWOS))
        rep_lo = simulate_thermal(topo_lo, power_per_die_mw=powers)

        topo_hi = ChipletTopology()
        topo_hi.add_die(ChipletDie(die_id=0))
        topo_hi.add_die(ChipletDie(die_id=1))
        topo_hi.add_link(InterposerLink.from_tech(0, 1, InterposerTech.ORGANIC))
        rep_hi = simulate_thermal(topo_hi, power_per_die_mw=powers)

        # Cold die under low-R bond heats up MORE than under high-R bond.
        assert rep_lo.die_temps[1] > rep_hi.die_temps[1], (
            f"COWOS coupling should heat die 1 more than ORGANIC: "
            f"COWOS={rep_lo.die_temps[1]:.2f} ORGANIC={rep_hi.die_temps[1]:.2f}"
        )

    def test_transient_converges_to_steady_state(self):
        """T(t→∞) of the implicit-Euler integrator → steady-state solution.

        After enough thermal time constants the transient must
        approach the directly-solved steady state to high precision.
        """
        topo = ChipletTopology.ring(4)
        powers = {i: 500.0 for i in range(4)}  # 0.5 W each
        rep = simulate_thermal(
            topo,
            power_per_die_mw=powers,
            transient_steps=2000,  # 2 s with default dt=1 ms
            transient_dt_s=1e-3,
        )
        # Final transient temperatures should match steady state to <0.01 °C.
        final = rep.transient_temps[-1]
        steady = np.array([rep.die_temps[d.die_id] for d in topo.dies])
        np.testing.assert_allclose(final, steady, atol=0.01)

    def test_transient_starts_at_ambient(self):
        """First time step starts from ambient (cold-boot transient)."""
        topo = ChipletTopology.ring(2)
        rep = simulate_thermal(
            topo,
            power_per_die_mw={0: 100.0, 1: 100.0},
            ambient_c=20.0,
            transient_steps=10,
        )
        # First step should be only slightly above ambient.
        first = rep.transient_temps[0]
        assert all(20.0 < t < 25.0 for t in first), (
            f"first transient step not near ambient: {first}"
        )


# ── Adaptive Routing Tests ───────────────────────────────────────────


class TestAdaptiveRouting:
    def test_no_congestion(self):
        topo = ChipletTopology.mesh_2d(2, 3)
        cong = CongestionReport()
        path = adaptive_route(topo, 0, 5, cong)
        assert path is not None
        assert path[0] == 0 and path[-1] == 5

    def test_avoids_congested_link(self):
        topo = make_torus(2, 3)
        cong = CongestionReport(utilisation={(0, 1): 0.95})
        path = adaptive_route(topo, 0, 1, cong, congestion_threshold=0.8)
        assert path is not None
        assert path[-1] == 1
        # Should not take the direct 0→1 since it's congested
        edges = list(zip(path[:-1], path[1:]))
        assert (0, 1) not in edges


# ── Link Protection Tests ────────────────────────────────────────────


class TestLinkProtection:
    def test_crc32_overhead(self):
        lp = LinkProtection(mode="crc32")
        assert lp.overhead_bits == 32

    def test_none_overhead(self):
        lp = LinkProtection(mode="none")
        assert lp.overhead_bits == 0
        assert lp.effective_bandwidth_ratio == 1.0

    def test_bandwidth_ratio(self):
        lp = LinkProtection(mode="crc32")
        assert lp.effective_bandwidth_ratio < 1.0
        assert lp.effective_bandwidth_ratio == 64.0 / 96.0

    def test_crc32_sv_output(self):
        sv = emit_crc32_sv(64)
        assert "sc_chiplet_crc32" in sv
        assert "SPDX" in sv
        assert "crc_error" in sv


# ── Bandwidth-Aware Routing Tests ────────────────────────────────────


class TestBandwidthAwareRouting:
    def test_same_die(self):
        topo = ChipletTopology.ring(3)
        path = bandwidth_aware_route(topo, 0, 0, 100.0)
        assert path == [0]

    def test_sufficient_bandwidth(self):
        topo = ChipletTopology.ring(3, InterposerTech.COWOS)
        path = bandwidth_aware_route(topo, 0, 1, 50.0)
        assert path is not None

    def test_insufficient_bandwidth(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(0))
        topo.add_die(ChipletDie(1))
        topo.add_link(InterposerLink(0, 1, bandwidth_gbps=1.0))
        path = bandwidth_aware_route(topo, 0, 1, 100.0)
        assert path is None


# ── Credit Config Tests ──────────────────────────────────────────────


class TestCreditConfig:
    def test_buffer_flits(self):
        cc = CreditConfig(initial_credits=16, credit_granularity=2)
        assert cc.buffer_flits == 32

    def test_credit_controller_sv(self):
        cc = CreditConfig(initial_credits=8)
        sv = emit_credit_controller_sv(cc, "test_link")
        assert "sc_chiplet_credit_test_link" in sv
        assert "INIT_CREDITS = 8" in sv
        assert "SPDX" in sv


# ── 3D Stacking Tests ────────────────────────────────────────────────


class Test3DStacking:
    def test_tsv_link_latency(self):
        tsv = TSVLink(src_die=0, dst_die=1, latency_ps=50.0)
        assert tsv.latency_ns == 0.05

    def test_tsv_bandwidth(self):
        tsv = TSVLink(src_die=0, dst_die=1, tsv_count=1024)
        assert tsv.bandwidth_gbps > 100

    def test_add_3d_stack(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(0))
        topo.add_die(ChipletDie(1))
        link = add_3d_stack(topo, 0, 1, StackingType.TSV_3D)
        assert len(topo.links) == 2  # bidirectional
        assert link.latency_ns < 0.1

    def test_hybrid_bonding(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(0))
        topo.add_die(ChipletDie(1))
        link = add_3d_stack(topo, 0, 1, StackingType.HYBRID_BONDING)
        assert link.bandwidth_gbps >= 512.0

    def test_3d_timing(self):
        topo = ChipletTopology()
        topo.add_die(ChipletDie(0))
        topo.add_die(ChipletDie(1))
        add_3d_stack(topo, 0, 1)
        result = simulate_timing(topo, 0, 1)
        assert result is not None
        assert result.total_latency_ns < 0.1


# ── Power Domain Tests ───────────────────────────────────────────────


class TestPowerDomain:
    def test_domain_active(self):
        pd = PowerDomain(domain_id=0, die_ids=[0, 1], is_active=True)
        assert not pd.is_gated

    def test_domain_gated(self):
        pd = PowerDomain(domain_id=0, die_ids=[0, 1], is_active=False)
        assert pd.is_gated

    def test_power_domain_map(self):
        pdm = PowerDomainMap()
        pdm.add_domain(PowerDomain(0, [0, 1], is_active=True))
        pdm.add_domain(PowerDomain(1, [2, 3], is_active=False))
        assert pdm.active_dies() == [0, 1]
        assert pdm.gated_dies() == [2, 3]

    def test_domain_for_die(self):
        pdm = PowerDomainMap()
        pdm.add_domain(PowerDomain(0, [0, 1]))
        pdm.add_domain(PowerDomain(1, [2, 3]))
        assert pdm.domain_for_die(2).domain_id == 1
        assert pdm.domain_for_die(99) is None

    def test_power_gating_sv(self):
        pd = PowerDomain(domain_id=0, die_ids=[0, 1], voltage_mv=750)
        sv = emit_power_gating_sv(pd)
        assert "sc_chiplet_pwr_domain_0" in sv
        assert "750 mV" in sv
        assert "SPDX" in sv


# ── Auto-Partitioning Tests ──────────────────────────────────────────


class TestPartitionAssignment:
    def test_assign_and_query(self):
        pa = PartitionAssignment()
        pa.assign(0, 0)
        pa.assign(1, 0)
        pa.assign(2, 1)
        assert pa.neurons_on_die(0) == [0, 1]
        assert pa.neurons_on_die(1) == [2]

    def test_to_routing_tables_local(self):
        pa = PartitionAssignment()
        pa.assign(0, 0)
        pa.assign(1, 0)
        # Both neurons on same die — no routing needed
        tables = pa.to_routing_tables([(0, 1, 256)])
        assert len(tables) == 0

    def test_to_routing_tables_cross_die(self):
        pa = PartitionAssignment()
        pa.assign(0, 0)
        pa.assign(1, 1)
        tables = pa.to_routing_tables([(0, 1, 256)])
        assert 0 in tables
        assert tables[0].num_entries == 1
        assert tables[0].entries[0].dst_die == 1

    def test_routing_table_from_partition(self):
        pa = PartitionAssignment()
        for i in range(4):
            pa.assign(i, 0)
        for i in range(4, 8):
            pa.assign(i, 1)
        connectivity = [
            (0, 4, 256),  # cross-die
            (1, 5, 128),  # cross-die
            (0, 1, 256),  # same die — ignored
            (4, 5, 256),  # same die — ignored
        ]
        tables = pa.to_routing_tables(connectivity)
        assert 0 in tables
        assert tables[0].num_entries == 2
        assert 1 not in tables  # no outgoing cross-die from die 1
