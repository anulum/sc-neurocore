# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet clock-crossing link bridges and routing-table RTL

"""Emit the clock-crossing bridge of each interposer link and each die's routing table.

A bridge writes its asynchronous FIFO on the source die's clock and runs its latency
pipeline on the destination die's clock, so the link latency is counted in cycles of the
destination clock. The FIFO is sized for the link jitter in cycles of the faster of the two
clocks. At the historical 200 MHz on both sides these equal the link's own
``latency_cycles`` and ``fifo_depth_log2``.
"""

from __future__ import annotations

import textwrap

from sc_neurocore.chiplet._sv import SPDX_HEADER, sv_signed
from sc_neurocore.chiplet.routing import RoutingEntry, RoutingTable
from sc_neurocore.chiplet.topology import ChipletDie, InterposerLink


def _cycles(ns: float, clock_mhz: float) -> int:
    """Return ``ns`` in whole cycles of a ``clock_mhz`` clock, at least one."""
    return max(1, int(ns * clock_mhz / 1000.0 + 0.5))


def bridge_timing(link: InterposerLink, src: ChipletDie, dst: ChipletDie) -> tuple[int, int]:
    """Return a bridge's latency in destination-clock cycles and its FIFO depth exponent.

    Parameters
    ----------
    link
        The interposer link the bridge implements.
    src, dst
        The link's source and destination dies, whose clocks the bridge crosses.

    Returns
    -------
    tuple[int, int]
        ``(latency_cycles, fifo_depth_log2)``: the link latency in cycles of ``dst``'s clock,
        and the smallest exponent, at least 3, whose depth holds four times the link jitter in
        cycles of the faster clock.
    """
    jitter_cycles = _cycles(link.jitter_ns, max(src.clock_mhz, dst.clock_mhz))
    depth = 3
    while (1 << depth) < jitter_cycles * 4:
        depth += 1
    return _cycles(link.latency_ns, dst.clock_mhz), depth


def emit_bridge(
    link: InterposerLink,
    src: ChipletDie,
    dst: ChipletDie,
    decorrelation_seed: int,
    decorrelate_links: bool = False,
) -> str:
    """Return the clock-crossing bridge of one link.

    Parameters
    ----------
    link
        The interposer link.
    src, dst
        The link's source and destination dies.
    decorrelation_seed
        Shared 16-bit LFSR seed of the link's payload scrambler.
    decorrelate_links
        Scramble the payload in flight when ``True``.

    Returns
    -------
    str
        SystemVerilog source of ``sc_chiplet_bridge_<src>_to_<dst>``.
    """
    latency, fifo_depth = bridge_timing(link, src, dst)
    decor_flag = 1 if decorrelate_links else 0
    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Bridge die {link.src_die} → die {link.dst_die} ({link.technology.value})
// Latency: {link.latency_ns} ns ({latency} cycles of die {dst.die_id}'s {dst.clock_mhz} MHz clock)
// Bandwidth: {link.bandwidth_gbps} Gb/s, BER: {link.bit_error_rate}
// Clocks: writes on die {src.die_id}'s clock (src_clk), reads on die {dst.die_id}'s (dst_clk)
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
        else if (s_tvalid && s_tready)
            tx_lfsr <= {{tx_lfsr[14:0], tx_lfsr[15] ^ tx_lfsr[13] ^ tx_lfsr[12] ^ tx_lfsr[10]}};
    wire [DATA_W-1:0] tx_scrambled, fifo_wr_data, fifo_rd_data, rx_descrambled;
    assign fifo_wr_data = (DECORRELATE_PAYLOAD != 0) ? tx_scrambled : s_tdata;
    reg [DATA_W-1:0] delay_pipe [0:LATENCY_CYC-1]; reg [LATENCY_CYC-1:0] valid_pipe;
    wire pipe_advance = !valid_pipe[LATENCY_CYC-1] || m_tready;
    sc_async_fifo #(.DATA_WIDTH(DATA_W), .DEPTH_LOG2(FIFO_DEPTH)) cdc_fifo (
        .wr_clk(src_clk), .wr_rst(src_rst), .wr_data(fifo_wr_data),
        .wr_en(s_tvalid && s_tready), .wr_full(fifo_wr_full),
        .rd_clk(dst_clk), .rd_rst(dst_rst), .rd_data(fifo_rd_data),
        .rd_en(!fifo_rd_empty && pipe_advance), .rd_empty(fifo_rd_empty)
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
        else if (m_tvalid && m_tready)
            rx_lfsr <= {{rx_lfsr[14:0], rx_lfsr[15] ^ rx_lfsr[13] ^ rx_lfsr[12] ^ rx_lfsr[10]}};
    genvar g;
    generate for (g = 0; g < DATA_W; g = g + 1) begin : scram
        assign tx_scrambled[g] = s_tdata[g] ^ tx_lfsr[g % 16];
        assign rx_descrambled[g] = delay_pipe[LATENCY_CYC-1][g] ^ rx_lfsr[g % 16];
    end endgenerate
    assign m_tdata = (DECORRELATE_PAYLOAD != 0) ? rx_descrambled : delay_pipe[LATENCY_CYC-1];
endmodule
""")


def emit_routing_table(table: RoutingTable, die: ChipletDie, die_id_w: int, pkg_aer_w: int) -> str:
    """Return one die's first-route lookup table.

    Parameters
    ----------
    table
        The die's routing table; the first entry per source neuron is kept.
    die
        The die the table belongs to.
    die_id_w, pkg_aer_w
        Package die-id and AER-id widths.

    Returns
    -------
    str
        SystemVerilog source of ``sc_chiplet_rtable_<die>``.
    """
    first_by_src: dict[int, RoutingEntry] = {}
    for entry in table.entries:
        first_by_src.setdefault(entry.src_neuron, entry)
    entries = [
        f"        rt_valid[{src}] = 1'b1; rt_target_die[{src}] = {die_id_w}'d{item.dst_die};\n"
        f"        rt_target_neuron[{src}] = {pkg_aer_w}'d{item.dst_neuron}; "
        f"rt_weight[{src}] = {sv_signed(item.weight_q88, die.data_width)};"
        for src, item in sorted(first_by_src.items())
    ]
    init_block = "\n".join(entries) if entries else "        // No inter-die routes"
    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Routing table for die {die.die_id}
module sc_chiplet_rtable_{die.die_id} #(
    parameter N_ENTRIES = {die.n_neurons}, parameter AER_ID_W = {die.aer_id_width},
    parameter PKG_AER_W = {pkg_aer_w}, parameter DIE_ID_W = {die_id_w},
    parameter DATA_W = {die.data_width}
)(
    input  wire clk, input  wire rst_n, input  wire [AER_ID_W-1:0] query_neuron,
    output reg  route_valid, output reg [DIE_ID_W-1:0] target_die,
    output reg  [PKG_AER_W-1:0] target_neuron, output reg signed [DATA_W-1:0] weight
);
    reg rt_valid [0:N_ENTRIES-1]; reg [DIE_ID_W-1:0] rt_target_die [0:N_ENTRIES-1];
    reg [PKG_AER_W-1:0] rt_target_neuron [0:N_ENTRIES-1];
    reg signed [DATA_W-1:0] rt_weight [0:N_ENTRIES-1];
    integer idx;
    initial begin
        for (idx = 0; idx < N_ENTRIES; idx = idx + 1) begin
            rt_valid[idx] = 1'b0; rt_target_die[idx] = 0;
            rt_target_neuron[idx] = 0; rt_weight[idx] = 0;
        end
{init_block}
    end
    always @(posedge clk) if (!rst_n) begin
        route_valid <= 1'b0; target_die <= 0; target_neuron <= 0; weight <= 0;
    end else begin
        route_valid <= rt_valid[query_neuron]; target_die <= rt_target_die[query_neuron];
        target_neuron <= rt_target_neuron[query_neuron]; weight <= rt_weight[query_neuron];
    end
endmodule
""")


__all__ = ["bridge_timing", "emit_bridge", "emit_routing_table"]
