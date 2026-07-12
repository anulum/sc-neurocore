# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet RTL generation contracts

"""End-to-end tests for connected die, bridge, route, top, and XDC emission."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    ChipletGenerator,
    ChipletOutput,
    ChipletTopology,
    InterposerTech,
    RoutingTable,
)


def _generated_ring(technology: InterposerTech = InterposerTech.UCIE) -> ChipletOutput:
    return ChipletGenerator().generate(ChipletTopology.ring(2, technology))


def test_generation_returns_complete_named_artefact_set() -> None:
    output = _generated_ring()
    assert isinstance(output, ChipletOutput)
    assert output.filelist == list(output.to_dict())
    assert "sc_chiplet_top.sv" in output.filelist
    assert "chiplet_constraints.xdc" in output.filelist


def test_top_connects_die_and_bridge_stream_ports() -> None:
    top = _generated_ring().top_sv
    assert "module sc_chiplet_top" in top
    assert "Dies: 2" in top
    assert ".link_out_1_tdata(link_0_1_tdata)" in top
    assert ".s_tdata(link_0_1_tdata)" in top
    assert ".m_tdata(link_0_1_rx_tdata)" in top
    assert "TODO" not in top


def test_die_wrapper_drives_links_and_instantiates_aer_router() -> None:
    die = _generated_ring().die_modules[0]
    assert "sc_aer_router" in die
    assert "lfsr" in die
    assert "assign link_out_1_tvalid = local_out_valid;" in die
    assert "assign link_in_1_tready = 1'b1;" in die
    assert "{{(64-AER_ID_W){1'b0}}, local_out_id}" in die


def test_bridge_contains_cdc_latency_and_decorrelation() -> None:
    bridge = _generated_ring(InterposerTech.EMIB).link_bridges[(0, 1)]
    assert "sc_async_fifo" in bridge
    assert "delay_pipe" in bridge
    assert "LATENCY_CYC" in bridge
    assert "decorrelated" in bridge
    assert "EMIB" in bridge


def test_routing_table_and_constraints_are_emitted() -> None:
    topology = ChipletTopology.ring(2)
    table = RoutingTable(die_id=0)
    table.add_route(5, 1, 10, 256)
    output = ChipletGenerator().generate(topology, routing={0: table})
    assert "rt_target_die" in output.routing_tables[0]
    assert "create_clock" in output.constraints_xdc
    assert "set_max_delay" in output.constraints_xdc


def test_every_generated_systemverilog_file_has_spdx_header() -> None:
    for name, content in _generated_ring().to_dict().items():
        if name.endswith(".sv"):
            assert "SPDX-License-Identifier" in content, name
