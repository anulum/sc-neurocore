# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet topology contracts

"""Behaviour and validation tests for package topology construction."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletTopology,
    InterposerLink,
    InterposerTech,
    StackingType,
    TSVLink,
    add_3d_stack,
    make_torus,
    simulate_timing,
)


class TestInterposerLink:
    """Interposer presets and fail-closed numerical contracts."""

    @pytest.mark.parametrize("technology", list(InterposerTech))
    def test_all_presets_are_physical(self, technology: InterposerTech) -> None:
        link = InterposerLink.from_tech(0, 1, technology)
        assert link.latency_ns >= 0
        assert link.bandwidth_gbps > 0
        assert 0 <= link.bit_error_rate <= 1

    def test_cowos_is_faster_than_organic(self) -> None:
        cowos = InterposerLink.from_tech(0, 1, InterposerTech.COWOS)
        organic = InterposerLink.from_tech(0, 1, InterposerTech.ORGANIC)
        assert cowos.latency_ns < organic.latency_ns
        assert cowos.bandwidth_gbps > organic.bandwidth_gbps

    def test_latency_and_jitter_determine_cycle_and_fifo_bounds(self) -> None:
        link = InterposerLink(src_die=0, dst_die=1, latency_ns=10.0, jitter_ns=5.0)
        assert link.latency_cycles == 2
        assert link.fifo_depth_log2 >= 3

    @pytest.mark.parametrize(
        ("constructor", "message"),
        [
            (lambda: InterposerLink(-1, 1), "src_die"),
            (lambda: InterposerLink(0, 1, latency_ns=math.nan), "latency_ns"),
            (lambda: InterposerLink(0, 1, jitter_ns=-1.0), "jitter_ns"),
            (lambda: InterposerLink(0, 1, bandwidth_gbps=0.0), "bandwidth_gbps"),
            (lambda: InterposerLink(0, 1, bit_error_rate=1.1), "bit_error_rate"),
            (lambda: InterposerLink(0, 1, data_width=0), "data_width"),
            (
                lambda: InterposerLink(0, 1, thermal_resistance_k_per_w=0.0),
                "thermal_resistance",
            ),
        ],
    )
    def test_invalid_link_contracts_fail(
        self, constructor: Callable[[], object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            constructor()


class TestChipletDie:
    """Die timing, seed, and width contracts."""

    def test_defaults_and_clock_period(self) -> None:
        die = ChipletDie(die_id=0, clock_mhz=100.0)
        assert die.clock_period_ns == 10.0
        assert die.n_neurons == 128

    def test_custom_seed_is_preserved(self) -> None:
        assert ChipletDie(die_id=5, lfsr_seed=0xBEEF).lfsr_seed == 0xBEEF

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: ChipletDie(-1),
            lambda: ChipletDie(0, clock_mhz=0.0),
            lambda: ChipletDie(0, lfsr_seed=0),
            lambda: ChipletDie(0, n_neurons=0),
        ],
    )
    def test_invalid_die_contracts_fail(self, constructor: Callable[[], object]) -> None:
        with pytest.raises(ValueError):
            constructor()


class TestPlanarTopologies:
    """Mesh, ring, star, and torus graph contracts."""

    def test_mesh_has_expected_dies_links_and_unique_seeds(self) -> None:
        topology = ChipletTopology.mesh_2d(2, 3)
        assert topology.num_dies == 6
        assert len(topology.links) == 7
        assert len({die.lfsr_seed for die in topology.dies}) == 6

    def test_ring_lookup_contracts(self) -> None:
        topology = ChipletTopology.ring(4, InterposerTech.EMIB)
        assert len(topology.links) == 4
        assert topology.get_links_from(0)[0].dst_die == 1
        assert topology.get_links_to(0)[0].src_die == 3
        assert topology.get_die(3) is not None
        assert topology.get_die(99) is None

    def test_star_is_bidirectional_through_hub(self) -> None:
        topology = ChipletTopology.star(5)
        assert len(topology.links) == 8
        assert len(topology.get_links_from(0)) == 4
        assert len(topology.get_links_to(0)) == 4
        timing = simulate_timing(topology, 1, 2)
        assert timing is not None and timing.path == [1, 0, 2]

    def test_torus_wraps_and_remains_connected(self) -> None:
        topology = make_torus(2, 3)
        assert topology.num_dies == 6
        assert len(topology.links) == 12
        assert 0 in {link.dst_die for link in topology.get_links_from(2)}
        assert simulate_timing(topology, 0, 5) is not None

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: ChipletTopology.mesh_2d(0, 1),
            lambda: ChipletTopology.ring(0),
            lambda: ChipletTopology.star(0),
            lambda: make_torus(1, 0),
        ],
    )
    def test_empty_topology_factories_fail(self, factory: Callable[[], object]) -> None:
        with pytest.raises(ValueError):
            factory()


class TestVerticalStacking:
    """TSV metadata and reciprocal topology insertion."""

    def test_tsv_unit_conversions(self) -> None:
        link = TSVLink(src_die=0, dst_die=1, tsv_count=1024, latency_ps=50.0)
        assert link.latency_ns == 0.05
        assert link.bandwidth_gbps > 100

    @pytest.mark.parametrize(
        ("stacking", "minimum_bandwidth"),
        [(StackingType.TSV_3D, 256.0), (StackingType.HYBRID_BONDING, 512.0)],
    )
    def test_stack_adds_reciprocal_links(
        self, stacking: StackingType, minimum_bandwidth: float
    ) -> None:
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
        link = add_3d_stack(topology, 0, 1, stacking)
        assert len(topology.links) == 2
        assert link.bandwidth_gbps >= minimum_bandwidth
        timing = simulate_timing(topology, 0, 1)
        assert timing is not None and timing.total_latency_ns < 0.1
