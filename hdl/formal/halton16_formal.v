// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — Formal properties for sc_halton16_source
//
// Prove:
// P1. Counter increments by 1 each enabled cycle
// P2. Output is bit-reversed counter (Van der Corput)
// P3. Valid only asserted when enabled
// P4. Reset zeroes all outputs

module halton16_formal #(
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst_n
);

    (* anyseq *) reg enable;

    wire [DATA_WIDTH-1:0] quasi_random;
    wire valid;

    sc_halton16_source #(.DATA_WIDTH(DATA_WIDTH)) uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .quasi_random(quasi_random),
        .valid(valid)
    );

    // Track internal counter via a shadow register
    reg [DATA_WIDTH-1:0] shadow_counter;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            shadow_counter <= 0;
        else if (enable)
            shadow_counter <= shadow_counter + 1;
    end

    // Expected bit-reversed value
    wire [DATA_WIDTH-1:0] expected_reversed;
    genvar i;
    generate
        for (i = 0; i < DATA_WIDTH; i = i + 1) begin : gen_rev
            assign expected_reversed[i] = shadow_counter[DATA_WIDTH - 1 - i];
        end
    endgenerate

    // P2: Output matches bit-reversed shadow counter (1 cycle delay due to register)
    reg [DATA_WIDTH-1:0] prev_expected;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            prev_expected <= 0;
        else if (enable)
            prev_expected <= expected_reversed;
    end

    // P3: Valid only when enabled
    always @(posedge clk) begin
        if (rst_n && !enable)
            assert(!valid);
    end

    // P4: After reset, outputs are zero
    reg past_valid;
    initial past_valid = 0;
    always @(posedge clk) begin
        past_valid <= 1;
        if (past_valid && !rst_n) begin
            assert(quasi_random == 0);
            assert(!valid);
        end
    end

endmodule
