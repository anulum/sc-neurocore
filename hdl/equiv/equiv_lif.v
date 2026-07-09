// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Miter circuit for LIF equivalence proof
//
// Drives both the DUT (sc_lif_neuron with REFRACTORY_PERIOD=0) and the
// reference model with identical inputs. Asserts outputs match at every
// cycle. SymbiYosys BMC proves this for ALL input sequences up to depth N.

`timescale 1ns / 1ps

module equiv_lif;

    parameter integer DATA_WIDTH = 16;
    parameter integer FRACTION = 8;
    parameter signed [DATA_WIDTH-1:0] V_REST      = 0;
    parameter signed [DATA_WIDTH-1:0] V_RESET     = 0;
    parameter signed [DATA_WIDTH-1:0] V_THRESHOLD = (1 << FRACTION);

    // Shared inputs (symbolic — SymbiYosys explores all values)
    reg clk = 0;
    reg rst_n;
    (* anyseq *) reg signed [DATA_WIDTH-1:0] leak_k;
    (* anyseq *) reg signed [DATA_WIDTH-1:0] gain_k;
    (* anyseq *) reg signed [DATA_WIDTH-1:0] I_t;
    (* anyseq *) reg signed [DATA_WIDTH-1:0] noise_in;

    // DUT outputs
    wire spike_dut, spike_ref;
    wire signed [DATA_WIDTH-1:0] v_dut, v_ref;

    // DUT: sc_lif_neuron with no refractory period
    sc_lif_neuron #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTION(FRACTION),
        .V_REST(V_REST),
        .V_RESET(V_RESET),
        .V_THRESHOLD(V_THRESHOLD),
        .REFRACTORY_PERIOD(0)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .leak_k(leak_k),
        .gain_k(gain_k),
        .I_t(I_t),
        .noise_in(noise_in),
        .spike_out(spike_dut),
        .v_out(v_dut)
    );

    // Reference model
    sc_lif_reference #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTION(FRACTION),
        .V_REST(V_REST),
        .V_RESET(V_RESET),
        .V_THRESHOLD(V_THRESHOLD)
    ) ref_model (
        .clk(clk),
        .rst_n(rst_n),
        .leak_k(leak_k),
        .gain_k(gain_k),
        .I_t(I_t),
        .noise_in(noise_in),
        .spike_out(spike_ref),
        .v_out(v_ref)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Reset protocol
    reg [3:0] cycle_count = 0;
    initial begin
        rst_n = 0;
    end

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (cycle_count == 2)
            rst_n <= 1;
    end

    // Equivalence assertions (active after reset)
    always @(posedge clk) begin
        if (rst_n) begin
            assert(spike_dut == spike_ref);
            assert(v_dut == v_ref);
        end
    end

endmodule
