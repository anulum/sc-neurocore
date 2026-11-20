# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet die wrapper RTL: local router, next-hop map, link arbitration

"""Emit one die's wrapper: its AER router, fan-out tables, next-hop map and link arbiter.

The wrapper runs entirely on the die's own clock; the bridges at its link ports cross to
the neighbouring dies' clocks.
"""

from __future__ import annotations

import textwrap

from sc_neurocore.chiplet._sv import SPDX_HEADER, sv_signed
from sc_neurocore.chiplet.routing import RoutingEntry, RoutingTable
from sc_neurocore.chiplet.topology import ChipletDie, ChipletTopology


def _route_inits(
    grouped: dict[int, list[RoutingEntry]], die: ChipletDie, die_id_w: int, pkg_aer_w: int
) -> str:
    lines: list[str] = []
    for src_n, entries in sorted(grouped.items()):
        lines.append(f"        rt_fanout_count[{src_n}] = {len(entries)};")
        for slot, item in enumerate(entries):
            target = f"{{{die_id_w}'d{item.dst_die}, {pkg_aer_w}'d{item.dst_neuron}}}"
            lines.append(
                f"        rt_target_id[{src_n}][{slot}] = {target};\n"
                f"        rt_target_weight[{src_n}][{slot}] = "
                f"{sv_signed(item.weight_q88, die.data_width)};"
            )
    return "\n".join(lines) if lines else "        // Empty route table"


def _arbiter(die: ChipletDie, topology: ChipletTopology, die_id_w: int, pkt_w: int) -> str:
    outgoing = topology.get_links_from(die.die_id)
    incoming = topology.get_links_to(die.die_id)
    lines = [
        "        local_taken = 1'b0; local_out_valid = 1'b0; router_ev_ready = 1'b0;",
        "        local_out_id = {AER_ID_W{1'b0}}; local_out_weight = {DATA_W{1'b0}};",
        *(
            f"        out_taken_{lk.dst_die} = 1'b0; link_out_{lk.dst_die}_tvalid = 1'b0; "
            f"link_out_{lk.dst_die}_tdata = {{{lk.data_width}{{1'b0}}}};"
            for lk in outgoing
        ),
        *(f"        link_in_{lk.src_die}_tready = 1'b0;" for lk in incoming),
    ]
    r_src = (
        "router_ev_valid",
        "router_ev_dest[DEST_W-1:PKG_AER_W]",
        "router_ev_dest[PKG_AER_W-1:0]",
        "router_ev_dest[AER_ID_W-1:0]",
        "router_weight_ext",
        "router_ev_weight",
        "router_ev_ready",
    )
    in_srcs = [
        (
            f"link_in_{s}_tvalid",
            f"link_in_{s}_tdata[PKT_W-1:PKG_DATA_W+PKG_AER_W]",
            f"link_in_{s}_tdata[PKG_AER_W-1:0]",
            f"link_in_{s}_tdata[AER_ID_W-1:0]",
            f"link_in_{s}_tdata[PKG_DATA_W+PKG_AER_W-1:PKG_AER_W]",
            f"$signed(link_in_{s}_tdata[PKG_AER_W +: DATA_W])",
            f"link_in_{s}_tready",
        )
        for lk in incoming
        for s in (lk.src_die,)
    ]
    for valid_s, dst_d, dst_n, dst_n_loc, w_pkg, w_loc, ready_s in [r_src, *in_srcs]:
        branches: list[str] = []
        for lk in outgoing:
            pad = lk.data_width - pkt_w
            fields = [f"{{{pad}{{1'b0}}}}"] if pad > 0 else []
            pkt = ",\n                        ".join([*fields, dst_d, w_pkg, dst_n])
            hop = f"{die_id_w}'d{lk.dst_die}"
            branches.append(
                f"                if (map_next_hop({dst_d}) == {hop}\n"
                f"                        && !out_taken_{lk.dst_die}) begin\n"
                f"                    out_taken_{lk.dst_die} = 1'b1;\n"
                f"                    link_out_{lk.dst_die}_tvalid = 1'b1;\n"
                f"                    link_out_{lk.dst_die}_tdata = {{\n"
                f"                        {pkt}}};\n"
                f"                    {ready_s} = link_out_{lk.dst_die}_tready;\n"
                f"                end"
            )
        else_body = (
            "\n".join(branches)
            if branches
            else f"                {ready_s} = 1'b1; // Isolated die: drain cross-die event"
        )
        lines.append(
            f"        if ({valid_s}) begin\n"
            f"            if ({dst_d} == DIE_ID[DIE_ID_W-1:0]) begin\n"
            f"                if (!local_taken) begin\n"
            f"                    local_taken = 1'b1; local_out_valid = 1'b1;\n"
            f"                    local_out_id = {dst_n_loc};\n"
            f"                    local_out_weight = {w_loc};\n"
            f"                    {ready_s} = local_out_ready;\n"
            f"                end\n"
            f"            end else begin\n{else_body}\n            end\n        end"
        )
    return "\n".join(lines)


def emit_die_wrapper(
    die: ChipletDie,
    topology: ChipletTopology,
    table: RoutingTable | None,
    next_hop_map: dict[int, int],
    die_id_w: int,
    pkg_aer_w: int,
    pkg_data_w: int,
) -> str:
    """Return ``sc_chiplet_die_<d>``: one die's router, tables, next-hop map and arbiter.

    Parameters
    ----------
    die
        The die to wrap.
    topology
        Validated die and link graph.
    table
        The die's routing table, or ``None`` for an empty one.
    next_hop_map
        Destination die to next-hop die, for every reachable destination.
    die_id_w, pkg_aer_w, pkg_data_w
        Package die-id, AER-id and data widths.

    Returns
    -------
    str
        SystemVerilog source of the die wrapper.
    """
    outgoing = topology.get_links_from(die.die_id)
    incoming = topology.get_links_to(die.die_id)
    grouped: dict[int, list[RoutingEntry]] = {}
    for entry in table.entries if table is not None else ():
        grouped.setdefault(entry.src_neuron, []).append(entry)
    max_fanout = max(2, 1 << max((len(v) for v in grouped.values()), default=1).bit_length())
    link_ports = "".join(
        f"    output reg [{lk.data_width - 1}:0] link_out_{lk.dst_die}_tdata,\n"
        f"    output reg link_out_{lk.dst_die}_tvalid, input wire link_out_{lk.dst_die}_tready,\n"
        for lk in outgoing
    ) + "".join(
        f"    input wire [{lk.data_width - 1}:0] link_in_{lk.src_die}_tdata,\n"
        f"    input wire link_in_{lk.src_die}_tvalid, output reg link_in_{lk.src_die}_tready,\n"
        for lk in incoming
    )
    hop_block = "\n".join(
        [
            *(
                f"                {die_id_w}'d{d}: map_next_hop = {die_id_w}'d{nh};"
                for d, nh in sorted(next_hop_map.items())
            ),
            "                default: map_next_hop = DIE_ID[DIE_ID_W-1:0];",
        ]
    )
    taken_decls = " ".join(
        ["reg local_taken;", *(f"reg out_taken_{lk.dst_die};" for lk in outgoing)]
    )
    w_ext = (
        "{{(PKG_DATA_W-DATA_W){router_ev_weight[DATA_W-1]}}, router_ev_weight}"
        if die.data_width < pkg_data_w
        else "router_ev_weight"
    )
    rt_init_block = _route_inits(grouped, die, die_id_w, pkg_aer_w)
    arb_block = _arbiter(die, topology, die_id_w, die_id_w + pkg_data_w + pkg_aer_w)
    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Die {die.die_id} wrapper
module sc_chiplet_die_{die.die_id} #(
    parameter DIE_ID = {die.die_id}, parameter N_NEURONS = {die.n_neurons},
    parameter AER_ID_W = {die.aer_id_width}, parameter DATA_W = {die.data_width},
    parameter DIE_ID_W = {die_id_w}, parameter PKG_AER_W = {pkg_aer_w},
    parameter PKG_DATA_W = {pkg_data_w}, parameter MAX_FANOUT = {max_fanout},
    parameter AER_PRIO_W = 2
)(
    input  wire clk, input  wire rst_n,
{link_ports}    input  wire local_spike_valid, output wire local_spike_ready,
    input  wire [AER_ID_W-1:0] local_spike_id,
    output reg  local_out_valid, input  wire local_out_ready,
    output reg  [AER_ID_W-1:0] local_out_id, output reg signed [DATA_W-1:0] local_out_weight
);
    localparam DEST_W = DIE_ID_W + PKG_AER_W, PKT_W = DIE_ID_W + PKG_DATA_W + PKG_AER_W;
    reg [$clog2(MAX_FANOUT)-1:0] rt_fanout_count [0:N_NEURONS-1];
    reg [DEST_W-1:0] rt_target_id [0:N_NEURONS-1][0:MAX_FANOUT-1];
    reg signed [DATA_W-1:0] rt_target_weight [0:N_NEURONS-1][0:MAX_FANOUT-1];
    integer init_s, init_f;
    initial begin
        for (init_s = 0; init_s < N_NEURONS; init_s = init_s + 1) begin
            rt_fanout_count[init_s] = 0;
            for (init_f = 0; init_f < MAX_FANOUT; init_f = init_f + 1) begin
                rt_target_id[init_s][init_f] = 0; rt_target_weight[init_s][init_f] = 0;
            end
        end
{rt_init_block}
    end
    wire router_ev_valid; reg router_ev_ready; wire [DEST_W-1:0] router_ev_dest;
    wire signed [DATA_W-1:0] router_ev_weight;
    wire signed [PKG_DATA_W-1:0] router_weight_ext = {w_ext};
    sc_aer_router #(
        .N_SRC(N_NEURONS), .N_TGT(N_NEURONS), .MAX_FANOUT(MAX_FANOUT),
        .NEURON_ID_WIDTH(DEST_W), .SRC_ID_WIDTH(AER_ID_W), .DATA_WIDTH(DATA_W),
        .PRIO_WIDTH(AER_PRIO_W)
    ) local_router (
        .clk(clk), .rst_n(rst_n),
        .in_event_valid(local_spike_valid), .in_event_ready(local_spike_ready),
        .in_neuron_id(local_spike_id), .in_timestamp(16'd0),
        .in_priority({{AER_PRIO_W{{1'b0}}}}), .fanout_count(rt_fanout_count),
        .target_id(rt_target_id), .target_weight(rt_target_weight),
        .out_event_valid(router_ev_valid), .out_event_ready(router_ev_ready),
        .out_target_id(router_ev_dest), .out_weight(router_ev_weight),
        .out_timestamp(), .out_priority(), .busy(), .queue_full(), .dropped_event(),
        .critical_deadline_violation()
    );
    function automatic [DIE_ID_W-1:0] map_next_hop(input [DIE_ID_W-1:0] dst);
        case (dst)
{hop_block}
        endcase
    endfunction
    {taken_decls}
    always @(*) begin
{arb_block}
    end
endmodule
""")


__all__ = ["emit_die_wrapper"]
