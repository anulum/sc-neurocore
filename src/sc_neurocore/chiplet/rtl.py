# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-die AER and AXI-Stream RTL generator

"""Emit die wrappers, CDC bridges, routing tables, package RTL, and XDC."""

from __future__ import annotations

from dataclasses import dataclass
import textwrap

from sc_neurocore.chiplet._sv import SPDX_HEADER
from sc_neurocore.chiplet.routing import (
    RoutingEntry,
    RoutingTable,
    compute_decorrelation_seeds,
    simulate_timing,
)
from sc_neurocore.chiplet.topology import ChipletDie, ChipletTopology, InterposerLink


def _sv_signed(value: int, width: int) -> str:
    return f"-{width}'sd{-value}" if value < 0 else f"{width}'sd{value}"


def _packet_widths(topology: ChipletTopology) -> tuple[int, int, int, int]:
    max_die_id = max((die.die_id for die in topology.dies), default=0)
    die_id_w = max(4, max_die_id.bit_length() + 1)
    pkg_aer_w = max((die.aer_id_width for die in topology.dies), default=10)
    pkg_data_w = max((die.data_width for die in topology.dies), default=16)
    return die_id_w, pkg_aer_w, pkg_data_w, die_id_w + pkg_data_w + pkg_aer_w


def _validate_inputs(
    topology: ChipletTopology,
    routing: dict[int, RoutingTable] | None,
    pkt_w: int,
) -> dict[int, dict[int, int]]:
    die_by_id: dict[int, ChipletDie] = {}
    for die in topology.dies:
        if die.die_id in die_by_id:
            raise ValueError(f"Duplicate die_id {die.die_id} in ChipletTopology")
        die_by_id[die.die_id] = die
    seen_links: set[tuple[int, int]] = set()
    for link in topology.links:
        if link.src_die not in die_by_id or link.dst_die not in die_by_id:
            raise ValueError(f"Link ({link.src_die} -> {link.dst_die}) references absent die_id")
        edge = (link.src_die, link.dst_die)
        if edge in seen_links:
            raise ValueError(f"Duplicate directed link ({link.src_die} -> {link.dst_die})")
        seen_links.add(edge)
        if link.data_width < pkt_w:
            raise ValueError(
                f"Link ({link.src_die} -> {link.dst_die}) data_width={link.data_width} < {pkt_w}"
            )
    next_hops: dict[int, dict[int, int]] = {die_id: {} for die_id in die_by_id}
    for src_id in die_by_id:
        for dst_id in die_by_id:
            if src_id != dst_id and (timed := simulate_timing(topology, src_id, dst_id)):
                next_hops[src_id][dst_id] = timed.path[1]
    if routing is not None:
        for die_id, table in routing.items():
            if die_id not in die_by_id or table.die_id != die_id:
                raise ValueError(f"Invalid routing table key or die_id for die {die_id}")
            src_die = die_by_id[die_id]
            src_lim = min(src_die.n_neurons, 1 << src_die.aer_id_width)
            for entry in table.entries:
                if not 0 <= entry.src_neuron < src_lim:
                    raise ValueError(f"src_neuron {entry.src_neuron} out of range on die {die_id}")
                if entry.dst_die not in die_by_id:
                    raise ValueError(f"dst_die {entry.dst_die} is absent from topology")
                dst_die = die_by_id[entry.dst_die]
                dst_lim = min(dst_die.n_neurons, 1 << dst_die.aer_id_width)
                eff_w = min(src_die.data_width, dst_die.data_width)
                if not 0 <= entry.dst_neuron < dst_lim:
                    raise ValueError(f"dst_neuron {entry.dst_neuron} out of range")
                if not -(1 << (eff_w - 1)) <= entry.weight_q88 <= (1 << (eff_w - 1)) - 1:
                    raise ValueError(f"weight_q88 {entry.weight_q88} out of signed range")
                if entry.dst_die != die_id and entry.dst_die not in next_hops[die_id]:
                    raise ValueError(f"No directed link path from die {die_id} to {entry.dst_die}")
    return next_hops


@dataclass
class ChipletOutput:
    """Contain generated source and constraint artefacts for one topology."""

    top_sv: str
    die_modules: dict[int, str]
    link_bridges: dict[tuple[int, int], str]
    routing_tables: dict[int, str]
    constraints_xdc: str
    filelist: list[str]

    def to_dict(self) -> dict[str, str]:
        """Return generated artefacts keyed by their output filename."""
        return {
            "sc_chiplet_top.sv": self.top_sv,
            **{f"sc_chiplet_die_{d}.sv": s for d, s in self.die_modules.items()},
            **{f"sc_chiplet_bridge_{u}_to_{v}.sv": s for (u, v), s in self.link_bridges.items()},
            **{f"sc_chiplet_rtable_{d}.sv": s for d, s in self.routing_tables.items()},
            "chiplet_constraints.xdc": self.constraints_xdc,
        }


class ChipletGenerator:
    """Generate connected multi-die routing RTL from a chiplet topology."""

    def generate(
        self,
        topology: ChipletTopology,
        routing: dict[int, RoutingTable] | None = None,
        *,
        decorrelate_links: bool = False,
    ) -> ChipletOutput:
        """Generate package RTL and timing constraints.

        Parameters
        ----------
        topology : ChipletTopology
            Directed die and interposer link graph. Per-die ``local_spike_id``
            inputs must stay strictly below ``die.n_neurons``.
        routing : dict[int, RoutingTable] | None, optional
            Optional per-die AER routing tables.
        decorrelate_links : bool, optional
            Scramble bridge ``s_tdata`` before the CDC FIFO and descramble after
            the latency pipeline from a shared per-link LFSR seed when ``True``
            (default ``False``), decorrelating in-flight interposer lanes while
            delivering bit-exact AER packets end to end.

        Returns
        -------
        ChipletOutput
            Generated top, die, bridge, route-table, and XDC artefacts.
        """
        die_id_w, pkg_aer_w, pkg_data_w, pkt_w = _packet_widths(topology)
        next_hops = _validate_inputs(topology, routing, pkt_w)
        seeds = compute_decorrelation_seeds(topology)
        die_modules: dict[int, str] = {}
        link_bridges: dict[tuple[int, int], str] = {}
        routing_tables: dict[int, str] = {}
        for die in topology.dies:
            table = routing.get(die.die_id) if routing else None
            die_modules[die.die_id] = self._emit_die_wrapper(
                die, topology, table, next_hops[die.die_id], die_id_w, pkg_aer_w, pkg_data_w
            )
            if table is not None:
                routing_tables[die.die_id] = self._emit_routing_table(
                    table, die, die_id_w, pkg_aer_w
                )
        for link in topology.links:
            seed = seeds.get((link.src_die, link.dst_die), 0xACE1)
            link_bridges[(link.src_die, link.dst_die)] = self._emit_bridge(
                link, seed, decorrelate_links
            )
        top, xdc = self._emit_top(topology), self._emit_constraints(topology)
        output = ChipletOutput(top, die_modules, link_bridges, routing_tables, xdc, [])
        output.filelist = list(output.to_dict())
        return output

    def _emit_die_wrapper(
        self,
        die: ChipletDie,
        topology: ChipletTopology,
        table: RoutingTable | None,
        next_hop_map: dict[int, int],
        die_id_w: int,
        pkg_aer_w: int,
        pkg_data_w: int,
    ) -> str:
        outgoing = topology.get_links_from(die.die_id)
        incoming = topology.get_links_to(die.die_id)
        grouped: dict[int, list[RoutingEntry]] = {}
        for entry in table.entries if table is not None else ():
            grouped.setdefault(entry.src_neuron, []).append(entry)
        max_fanout = max(2, 1 << max((len(v) for v in grouped.values()), default=1).bit_length())
        out_p = "".join(
            f"    output reg [{lk.data_width - 1}:0] link_out_{lk.dst_die}_tdata, "
            f"output reg link_out_{lk.dst_die}_tvalid, input wire link_out_{lk.dst_die}_tready,\n"
            for lk in outgoing
        )
        in_p = "".join(
            f"    input wire [{lk.data_width - 1}:0] link_in_{lk.src_die}_tdata, "
            f"input wire link_in_{lk.src_die}_tvalid, output reg link_in_{lk.src_die}_tready,\n"
            for lk in incoming
        )
        link_ports = out_p + in_p
        rt_inits: list[str] = []
        for src_n, entries in sorted(grouped.items()):
            rt_inits.append(f"        rt_fanout_count[{src_n}] = {len(entries)};")
            for slot, item in enumerate(entries):
                w_lit = _sv_signed(item.weight_q88, die.data_width)
                rt_inits.append(
                    f"        rt_target_id[{src_n}][{slot}] = {{{die_id_w}'d{item.dst_die}, {pkg_aer_w}'d{item.dst_neuron}}};\n"
                    f"        rt_target_weight[{src_n}][{slot}] = {w_lit};"
                )
        rt_init_block = "\n".join(rt_inits) if rt_inits else "        // Empty route table"
        hop_lines = [
            f"                {die_id_w}'d{d}: map_next_hop = {die_id_w}'d{nh};"
            for d, nh in sorted(next_hop_map.items())
        ]
        hop_block = "\n".join(
            [*hop_lines, "                default: map_next_hop = DIE_ID[DIE_ID_W-1:0];"]
        )
        arb_lines = [
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
                pad = lk.data_width - (die_id_w + pkg_data_w + pkg_aer_w)
                pkt = (
                    f"{{{{{pad}{{1'b0}}}}, {dst_d}, {w_pkg}, {dst_n}}}"
                    if pad > 0
                    else f"{{{dst_d}, {w_pkg}, {dst_n}}}"
                )
                branches.append(
                    f"                if (map_next_hop({dst_d}) == {die_id_w}'d{lk.dst_die} && !out_taken_{lk.dst_die}) begin\n"
                    f"                    out_taken_{lk.dst_die} = 1'b1; link_out_{lk.dst_die}_tvalid = 1'b1;\n"
                    f"                    link_out_{lk.dst_die}_tdata = {pkt}; {ready_s} = link_out_{lk.dst_die}_tready;\n"
                    f"                end"
                )
            else_body = (
                "\n".join(branches)
                if branches
                else f"                {ready_s} = 1'b1; // Isolated die: drain cross-die event"
            )
            arb_lines.append(
                f"        if ({valid_s}) begin\n"
                f"            if ({dst_d} == DIE_ID[DIE_ID_W-1:0]) begin\n"
                f"                if (!local_taken) begin\n"
                f"                    local_taken = 1'b1; local_out_valid = 1'b1;\n"
                f"                    local_out_id = {dst_n_loc}; local_out_weight = {w_loc}; {ready_s} = local_out_ready;\n"
                f"                end\n"
                f"            end else begin\n{else_body}\n            end\n        end"
            )
        taken_decls = " ".join(
            ["reg local_taken;", *(f"reg out_taken_{lk.dst_die};" for lk in outgoing)]
        )
        w_ext = (
            "{{(PKG_DATA_W-DATA_W){router_ev_weight[DATA_W-1]}}, router_ev_weight}"
            if die.data_width < pkg_data_w
            else "router_ev_weight"
        )
        arb_block = "\n".join(arb_lines)
        return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Die {die.die_id} wrapper
module sc_chiplet_die_{die.die_id} #(
    parameter DIE_ID = {die.die_id}, parameter N_NEURONS = {die.n_neurons}, parameter AER_ID_W = {die.aer_id_width},
    parameter DATA_W = {die.data_width}, parameter DIE_ID_W = {die_id_w}, parameter PKG_AER_W = {pkg_aer_w},
    parameter PKG_DATA_W = {pkg_data_w}, parameter MAX_FANOUT = {max_fanout}, parameter AER_PRIO_W = 2
)(
    input  wire clk, input  wire rst_n,
{link_ports}    input  wire local_spike_valid, output wire local_spike_ready, input  wire [AER_ID_W-1:0] local_spike_id,
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
    wire signed [DATA_W-1:0] router_ev_weight; wire signed [PKG_DATA_W-1:0] router_weight_ext = {w_ext};
    sc_aer_router #(.N_SRC(N_NEURONS), .N_TGT(N_NEURONS), .MAX_FANOUT(MAX_FANOUT), .NEURON_ID_WIDTH(DEST_W), .SRC_ID_WIDTH(AER_ID_W), .DATA_WIDTH(DATA_W), .PRIO_WIDTH(AER_PRIO_W)) local_router (
        .clk(clk), .rst_n(rst_n), .in_event_valid(local_spike_valid), .in_event_ready(local_spike_ready), .in_neuron_id(local_spike_id),
        .in_timestamp(16'd0), .in_priority({{AER_PRIO_W{{1'b0}}}}), .fanout_count(rt_fanout_count), .target_id(rt_target_id), .target_weight(rt_target_weight),
        .out_event_valid(router_ev_valid), .out_event_ready(router_ev_ready), .out_target_id(router_ev_dest), .out_weight(router_ev_weight),
        .out_timestamp(), .out_priority(), .busy(), .queue_full(), .dropped_event(), .critical_deadline_violation()
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

    def _emit_bridge(
        self, link: InterposerLink, decorrelation_seed: int, decorrelate_links: bool = False
    ) -> str:
        fifo_depth, latency = link.fifo_depth_log2, link.latency_cycles
        decor_flag = 1 if decorrelate_links else 0
        return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Bridge die {link.src_die} → die {link.dst_die} ({link.technology.value})
// Latency: {link.latency_ns} ns ({latency} cycles @ 200 MHz), Bandwidth: {link.bandwidth_gbps} Gb/s, BER: {link.bit_error_rate}
module sc_chiplet_bridge_{link.src_die}_to_{link.dst_die} #(
    parameter DATA_W = {link.data_width}, parameter FIFO_DEPTH = {fifo_depth},
    parameter LATENCY_CYC = {latency}, parameter DECOR_SEED = 16'h{decorrelation_seed:04X},
    parameter DECORRELATE_PAYLOAD = {decor_flag}
)(
    input  wire src_clk, input  wire src_rst, input  wire dst_clk, input  wire dst_rst,
    input  wire [DATA_W-1:0] s_tdata, input  wire s_tvalid, output wire s_tready,
    output wire [DATA_W-1:0] m_tdata, output wire m_tvalid, input  wire m_tready
);
    wire fifo_rd_empty, fifo_wr_full; assign s_tready = !fifo_wr_full;
    reg [15:0] tx_lfsr, rx_lfsr;
    always @(posedge src_clk) if (src_rst) tx_lfsr <= DECOR_SEED;
        else if (s_tvalid && s_tready) tx_lfsr <= {{tx_lfsr[14:0], tx_lfsr[15] ^ tx_lfsr[13] ^ tx_lfsr[12] ^ tx_lfsr[10]}};
    wire [DATA_W-1:0] tx_scrambled, fifo_wr_data, fifo_rd_data, rx_descrambled;
    assign fifo_wr_data = (DECORRELATE_PAYLOAD != 0) ? tx_scrambled : s_tdata;
    reg [DATA_W-1:0] delay_pipe [0:LATENCY_CYC-1]; reg [LATENCY_CYC-1:0] valid_pipe;
    wire pipe_advance = !valid_pipe[LATENCY_CYC-1] || m_tready;
    sc_async_fifo #(.DATA_WIDTH(DATA_W), .DEPTH_LOG2(FIFO_DEPTH)) cdc_fifo (
        .wr_clk(src_clk), .wr_rst(src_rst), .wr_data(fifo_wr_data), .wr_en(s_tvalid && s_tready), .wr_full(fifo_wr_full),
        .rd_clk(dst_clk), .rd_rst(dst_rst), .rd_data(fifo_rd_data), .rd_en(!fifo_rd_empty && pipe_advance), .rd_empty(fifo_rd_empty)
    );
    integer k;
    always @(posedge dst_clk) if (dst_rst) begin
        valid_pipe <= 0; for (k = 0; k < LATENCY_CYC; k = k + 1) delay_pipe[k] <= 0;
    end else if (pipe_advance) begin
        delay_pipe[0] <= fifo_rd_data; valid_pipe[0] <= !fifo_rd_empty;
        for (k = 1; k < LATENCY_CYC; k = k + 1) begin
            delay_pipe[k] <= delay_pipe[k-1]; valid_pipe[k] <= valid_pipe[k-1];
        end
    end
    assign m_tvalid = valid_pipe[LATENCY_CYC-1];
    always @(posedge dst_clk) if (dst_rst) rx_lfsr <= DECOR_SEED;
        else if (m_tvalid && m_tready) rx_lfsr <= {{rx_lfsr[14:0], rx_lfsr[15] ^ rx_lfsr[13] ^ rx_lfsr[12] ^ rx_lfsr[10]}};
    genvar g;
    generate for (g = 0; g < DATA_W; g = g + 1) begin : scram
        assign tx_scrambled[g] = s_tdata[g] ^ tx_lfsr[g % 16];
        assign rx_descrambled[g] = delay_pipe[LATENCY_CYC-1][g] ^ rx_lfsr[g % 16];
    end endgenerate
    assign m_tdata = (DECORRELATE_PAYLOAD != 0) ? rx_descrambled : delay_pipe[LATENCY_CYC-1];
endmodule
""")

    def _emit_routing_table(
        self, table: RoutingTable, die: ChipletDie, die_id_w: int, pkg_aer_w: int
    ) -> str:
        first_by_src: dict[int, RoutingEntry] = {}
        for entry in table.entries:
            first_by_src.setdefault(entry.src_neuron, entry)
        entries = [
            f"        rt_valid[{src}] = 1'b1; rt_target_die[{src}] = {die_id_w}'d{item.dst_die};\n"
            f"        rt_target_neuron[{src}] = {pkg_aer_w}'d{item.dst_neuron}; "
            f"rt_weight[{src}] = {_sv_signed(item.weight_q88, die.data_width)};"
            for src, item in sorted(first_by_src.items())
        ]
        init_block = "\n".join(entries) if entries else "        // No inter-die routes"
        return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Routing table for die {die.die_id}
module sc_chiplet_rtable_{die.die_id} #(
    parameter N_ENTRIES = {die.n_neurons}, parameter AER_ID_W = {die.aer_id_width},
    parameter PKG_AER_W = {pkg_aer_w}, parameter DIE_ID_W = {die_id_w}, parameter DATA_W = {die.data_width}
)(
    input  wire clk, input  wire rst_n, input  wire [AER_ID_W-1:0] query_neuron,
    output reg  route_valid, output reg [DIE_ID_W-1:0] target_die,
    output reg  [PKG_AER_W-1:0] target_neuron, output reg signed [DATA_W-1:0] weight
);
    reg rt_valid [0:N_ENTRIES-1]; reg [DIE_ID_W-1:0] rt_target_die [0:N_ENTRIES-1];
    reg [PKG_AER_W-1:0] rt_target_neuron [0:N_ENTRIES-1]; reg signed [DATA_W-1:0] rt_weight [0:N_ENTRIES-1];
    integer idx;
    initial begin
        for (idx = 0; idx < N_ENTRIES; idx = idx + 1) begin rt_valid[idx] = 1'b0; rt_target_die[idx] = 0; rt_target_neuron[idx] = 0; rt_weight[idx] = 0; end
{init_block}
    end
    always @(posedge clk) if (!rst_n) begin route_valid <= 1'b0; target_die <= 0; target_neuron <= 0; weight <= 0; end
        else begin route_valid <= rt_valid[query_neuron]; target_die <= rt_target_die[query_neuron]; target_neuron <= rt_target_neuron[query_neuron]; weight <= rt_weight[query_neuron]; end
endmodule
""")

    def _emit_top(self, topology: ChipletTopology) -> str:
        die_ports = [
            f"    input wire die_{d.die_id}_spike_valid, output wire die_{d.die_id}_spike_ready,\n"
            f"    input wire [{d.aer_id_width - 1}:0] die_{d.die_id}_spike_id, "
            f"output wire die_{d.die_id}_out_valid, input wire die_{d.die_id}_out_ready,\n"
            f"    output wire [{d.aer_id_width - 1}:0] die_{d.die_id}_out_id, "
            f"output wire signed [{d.data_width - 1}:0] die_{d.die_id}_out_weight"
            for d in topology.dies
        ]
        top_port_block = ",\n".join(["    input wire clk, input wire rst_n", *die_ports])
        wire_block = "\n".join(
            f"    wire [{lk.data_width - 1}:0] link_{lk.src_die}_{lk.dst_die}_tdata, link_{lk.src_die}_{lk.dst_die}_rx_tdata;\n"
            f"    wire link_{lk.src_die}_{lk.dst_die}_tvalid, link_{lk.src_die}_{lk.dst_die}_tready, "
            f"link_{lk.src_die}_{lk.dst_die}_rx_tvalid, link_{lk.src_die}_{lk.dst_die}_rx_tready;"
            for lk in topology.links
        )
        die_instances: list[str] = []
        for die in topology.dies:
            d = die.die_id
            out_c = [
                f"        .link_out_{lk.dst_die}_tdata(link_{lk.src_die}_{lk.dst_die}_tdata), "
                f".link_out_{lk.dst_die}_tvalid(link_{lk.src_die}_{lk.dst_die}_tvalid), "
                f".link_out_{lk.dst_die}_tready(link_{lk.src_die}_{lk.dst_die}_tready)"
                for lk in topology.get_links_from(d)
            ]
            in_c = [
                f"        .link_in_{lk.src_die}_tdata(link_{lk.src_die}_{lk.dst_die}_rx_tdata), "
                f".link_in_{lk.src_die}_tvalid(link_{lk.src_die}_{lk.dst_die}_rx_tvalid), "
                f".link_in_{lk.src_die}_tready(link_{lk.src_die}_{lk.dst_die}_rx_tready)"
                for lk in topology.get_links_to(d)
            ]
            conns = [*out_c, *in_c]
            link_block = ",\n" + ",\n".join(conns) if conns else ""
            die_instances.append(
                f"    sc_chiplet_die_{d} die_{d}_inst (\n"
                f"        .clk(clk), .rst_n(rst_n){link_block},\n"
                f"        .local_spike_valid(die_{d}_spike_valid), .local_spike_ready(die_{d}_spike_ready),\n"
                f"        .local_spike_id(die_{d}_spike_id), .local_out_valid(die_{d}_out_valid),\n"
                f"        .local_out_ready(die_{d}_out_ready), .local_out_id(die_{d}_out_id),\n"
                f"        .local_out_weight(die_{d}_out_weight)\n    );"
            )
        bridge_block = "\n\n".join(
            f"    sc_chiplet_bridge_{lk.src_die}_to_{lk.dst_die} bridge_{lk.src_die}_{lk.dst_die}_inst (\n"
            f"        .src_clk(clk), .src_rst(!rst_n), .dst_clk(clk), .dst_rst(!rst_n),\n"
            f"        .s_tdata(link_{lk.src_die}_{lk.dst_die}_tdata), .s_tvalid(link_{lk.src_die}_{lk.dst_die}_tvalid),\n"
            f"        .s_tready(link_{lk.src_die}_{lk.dst_die}_tready), .m_tdata(link_{lk.src_die}_{lk.dst_die}_rx_tdata),\n"
            f"        .m_tvalid(link_{lk.src_die}_{lk.dst_die}_rx_tvalid), .m_tready(link_{lk.src_die}_{lk.dst_die}_rx_tready)\n    );"
            for lk in topology.links
        )
        die_block = "\n\n".join(die_instances)
        return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Top-level multi-die package
// Dies: {topology.num_dies}
// Links: {len(topology.links)}
module sc_chiplet_top (
{top_port_block}
);
{wire_block}
{die_block}
{bridge_block}
endmodule
""")

    def _emit_constraints(self, topology: ChipletTopology) -> str:
        hdr = (
            "# SC-NeuroCore Chiplet — Timing constraints\n# Auto-generated for multi-die package\n"
        )
        dies = "\n".join(
            f"# Die {d.die_id}: {d.clock_mhz} MHz\n"
            f"create_clock -name clk_die_{d.die_id} -period {d.clock_period_ns:.3f} [get_pins die_{d.die_id}_inst/clk]"
            for d in topology.dies
        )
        links = "\n".join(
            f"# Link {lk.src_die} → {lk.dst_die}: {lk.latency_ns} ns\n"
            f"set_max_delay -datapath_only -from [get_clocks clk_die_{lk.src_die}] -to [get_clocks clk_die_{lk.dst_die}] {lk.latency_ns}"
            for lk in topology.links
        )
        return f"{hdr}\n{dies}\n\n{links}\n"


__all__ = ["ChipletGenerator", "ChipletOutput"]
