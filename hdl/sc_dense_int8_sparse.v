// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sparse int8 dense matvec (Masquelier SHD model)
//
// Per-tensor symmetric int8 dense layer with binary spike inputs and Q8.8
// output, matching tools/shd_q88_reference.py::sparse_dense_q88():
//
//   for j in 0..OUT_FEATURES-1:
//       accum[j] = sum_i W[j, i] * spike_in[i]            (signed int)
//       product   = accum[j] * scale_q16_16               (signed int)
//       v_q88     = product >>> 8                         (Q8.8 = product / 256)
//       out[j]    = saturate(v_q88, [-32768, 32767])
//
// Where `scale_q16_16` is the per-tensor quantisation scale rendered as
// Q16.16 (round(scale * 65536)). This keeps the multiply integer-only
// while preserving 16-bit fractional precision for the small per-tensor
// scales seen in practice (~1e-4 to 1e-1).
//
// Weights are loaded once at elaboration time from a flat $readmemh hex
// file (one 8-bit signed value per line, row-major order
// `[j*IN_FEATURES + i]`), exactly the format emitted by
// tools/extract_shd_weights.py::write_int8_hex(). The path is a string
// parameter — for cosim, the Python harness writes the file into a
// temporary working directory and runs `vvp` from there so that the
// default relative path "weights.hex" resolves.
//
// The compute path is fully combinational over the 1-bit input vector
// and registers the result on the next rising edge. For the SHD network
// (140 → 128 → 128 → 20) the resulting LUT cone is large but functional
// in iverilog and acceptable for first-pass synthesis on Zynq-7020. A
// streaming time-multiplexed variant is a future optimisation (#167).
//
// Verified by:
//   hdl/tb_sc_dense_int8_sparse.v
//   tools/cosim_dense_int8_verilog.py  (5 stimulus cases, bit-true match)

`timescale 1ns / 1ps

module sc_dense_int8_sparse #(
    parameter integer IN_FEATURES  = 140,
    parameter integer OUT_FEATURES = 128,
    parameter integer ACCUM_WIDTH  = 24,
    parameter         WEIGHT_FILE  = "weights.hex"
)(
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire signed [31:0]              scale_q16_16,
    input  wire [IN_FEATURES-1:0]          spikes_in,
    // Registered output bus (1-cycle latency, used by tb_sc_dense_int8_sparse).
    output reg  [OUT_FEATURES*16-1:0]      out_q88_packed,
    // Combinational output bus (no register, used by sc_shd_top so the
    // full network can advance one full step per clock cycle).
    output wire [OUT_FEATURES*16-1:0]      out_q88_packed_comb
);

    // ------------------------------------------------------------------
    // Weight ROM: row-major, j*IN_FEATURES + i
    // ------------------------------------------------------------------
    reg [7:0] weights [0:OUT_FEATURES*IN_FEATURES-1];

    // Weight initialisation: only performed during simulation. Synthesis
    // tools (yosys, Vivado) define `SYNTHESIS` so the `$readmemh` call is
    // skipped — the weight ROM becomes an uninitialised BRAM/LUT that the
    // real bitstream writes through AXI at boot time. The default
    // WEIGHT_FILE = "weights.hex" is only meaningful to simulation
    // testbenches that stage the file into their temporary work dir.
    initial begin
`ifndef SYNTHESIS
        $readmemh(WEIGHT_FILE, weights);
`endif
    end

    // ------------------------------------------------------------------
    // Combinational matvec + scale + saturate
    // ------------------------------------------------------------------
    integer j;
    integer i;
    reg signed [ACCUM_WIDTH-1:0] accum_comb [0:OUT_FEATURES-1];
    reg signed [55:0]            product_comb;
    reg signed [47:0]            shifted_comb;
    reg signed [15:0]            sat_comb [0:OUT_FEATURES-1];

    // Sign-extension helper for 8-bit weight to ACCUM_WIDTH-bit signed
    // (Verilog 2001 has no $signed of an array element with width
    // promotion — do it explicitly via concatenation of the sign bit.)
    function automatic signed [ACCUM_WIDTH-1:0] signext_w;
        input [7:0] w;
        begin
            signext_w = {{(ACCUM_WIDTH-8){w[7]}}, w};
        end
    endfunction

    always @(*) begin
        for (j = 0; j < OUT_FEATURES; j = j + 1) begin
            // 1) sparse signed accumulator over active spikes
            accum_comb[j] = {ACCUM_WIDTH{1'b0}};
            for (i = 0; i < IN_FEATURES; i = i + 1) begin
                if (spikes_in[i]) begin
                    accum_comb[j] = accum_comb[j]
                        + signext_w(weights[j*IN_FEATURES + i]);
                end
            end

            // 2) multiply by Q16.16 scale (signed × signed)
            product_comb = $signed(accum_comb[j]) * $signed(scale_q16_16);

            // 3) ASHR by 8 to recover Q8.8 (Verilog `>>>` on signed wires
            //    matches Python's `>>` on int — round toward -inf)
            shifted_comb = product_comb >>> 8;

            // 4) saturate to signed 16-bit Q8.8
            if (shifted_comb > 48'sd32767)
                sat_comb[j] = 16'sd32767;
            else if (shifted_comb < -48'sd32768)
                sat_comb[j] = -16'sd32768;
            else
                sat_comb[j] = shifted_comb[15:0];
        end
    end

    // Register the output bus on every rising edge.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_q88_packed <= {(OUT_FEATURES*16){1'b0}};
        end else begin
            for (j = 0; j < OUT_FEATURES; j = j + 1) begin
                out_q88_packed[16*j +: 16] <= sat_comb[j];
            end
        end
    end

    // Combinational tap of the same value (zero-latency view of `sat_comb`).
    genvar gj;
    generate
        for (gj = 0; gj < OUT_FEATURES; gj = gj + 1) begin: comb_pack
            assign out_q88_packed_comb[16*gj +: 16] = sat_comb[gj];
        end
    endgenerate

endmodule
