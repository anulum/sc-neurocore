# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyAndCongestion from former test_chiplet_routing.py

"""Focused suite: TestEnergyAndCongestion from former test_chiplet_routing.py."""

from __future__ import annotations

from chiplet_routing_support import *  # noqa: F403

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
