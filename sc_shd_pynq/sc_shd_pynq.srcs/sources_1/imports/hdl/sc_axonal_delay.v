// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Single-source axonal delay buffer (Masquelier SHD model)
//
// Per-source-neuron circular buffer of past spikes. Each source neuron has
// a fixed integer delay d ∈ [0, DEPTH-1] presented as `read_offset`. The
// effective delay equals `read_offset` cycles (read_offset=0 means
// "pass the current spike straight through this cycle").
//
// Per-step semantics — bit-true with tools/shd_q88_reference.py
// AxonalDelayBuffer.step():
//
//   1. buf[head] <- spike_in                       (latched at clock edge)
//   2. read_idx  = (head - read_offset) mod DEPTH  (combinational)
//   3. spike_out = buf[read_idx]  OR  spike_in if read_idx == head
//                                                  (combinational)
//   4. head <- (head + 1) mod DEPTH                (latched)
//
// The "passthrough" branch in step 3 is what makes a 0-delay tap return the
// spike that arrives this very cycle: in the Python model the write occurs
// before the read, so they observe the same spike; in the Verilog model the
// write is non-blocking and would not be visible to a same-cycle read of
// buf_reg, so we explicitly mux spike_in for the matching cell.
//
// Storage: DEPTH bits + ceil(log2(DEPTH))-bit head pointer per instance.
// For DEPTH=31 that is 31 LUT FFs + 5-bit counter — small enough to live
// in distributed RAM. For the SHD top wrapper, instantiate one module per
// source neuron (140 for layer 1, 128 for layer 2). A BRAM-backed
// multi-source variant is a future optimisation.
//
// Verified by:
//   hdl/tb_sc_axonal_delay.v
//   tools/cosim_axonal_delay_verilog.py  (5 stimulus cases, bit-true match)

`timescale 1ns / 1ps

module sc_axonal_delay #(
    parameter integer DEPTH     = 31,
    parameter integer PTR_WIDTH = 5    // ceil(log2(31)) = 5
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  spike_in,
    input  wire [PTR_WIDTH-1:0]  read_offset,  // 0..DEPTH-1
    output wire                  spike_out
);

    // Storage: depth-bit shift register + write pointer (head).
    reg [DEPTH-1:0]     buf_reg;
    reg [PTR_WIDTH-1:0] head;

    // ------------------------------------------------------------------
    // Combinational read index = (head - read_offset) mod DEPTH
    //
    // head, read_offset ∈ [0, DEPTH-1] so the unsigned subtraction can
    // underflow by at most (DEPTH-1). One conditional add of DEPTH is
    // sufficient to wrap back into [0, DEPTH-1].
    // ------------------------------------------------------------------
    wire signed [PTR_WIDTH+1:0] diff =
        $signed({2'b00, head}) - $signed({2'b00, read_offset});
    wire [PTR_WIDTH-1:0] read_idx =
        (diff < 0) ? (diff + DEPTH) : diff[PTR_WIDTH-1:0];

    // Passthrough mux: if read_idx == head, return the spike being written
    // this cycle (matches Python's write-before-read ordering).
    wire same_cell = (read_idx == head);
    assign spike_out = same_cell ? spike_in : buf_reg[read_idx];

    // ------------------------------------------------------------------
    // Sequential update: write current spike at head, advance head.
    // Variable bit-select on the LHS of a non-blocking assign is legal
    // standard Verilog (10.4.4 IEEE 1800).
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_reg <= {DEPTH{1'b0}};
            head    <= {PTR_WIDTH{1'b0}};
        end else begin
            buf_reg[head] <= spike_in;
            head <= (head == DEPTH - 1) ? {PTR_WIDTH{1'b0}} : head + 1'b1;
        end
    end

endmodule
