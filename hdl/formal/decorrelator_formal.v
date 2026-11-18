// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal properties for sc_decorrelator

// Three instances: the default one (8 streams of 16 bits), the one the
// Verilog generator and the testbench use (4 streams of 8 bits), and one
// whose width is not a power of two (2 streams of 12 bits), which takes the
// modulo branch of the rotation.
//
// For every source word and every clock (unbounded, task prove, k-induction):
// P0. The phase counter of each instance (read through the FORMAL-only port
//     formal_phase_counter) is 0 after reset and counts clocks since reset.
// P1. Each output word is the source word of the previous clock rotated left
//     by (i XOR phase XOR seed nibble i) modulo the width, where the phase
//     counts clocks since the last reset. The rotation is written here as
//     the upper half of the doubled word shifted left, not with the RTL's shifts.
// P2. Each output word keeps the source word's population count. This
//     follows from P1, since a rotation permutes the bits; it is not asserted
//     separately, because bit-level solvers prove popcount equality under a
//     symbolic rotation very slowly.
// P3. For the default parameters, a one-hot source word gives eight pairwise
//     distinct output words: the streams' rotations differ in every clock.
// P4. During reset every output word is zero.

module decorrelator_formal (
    input wire clk
);

    (* anyseq *) reg rst_n;
    (* anyseq *) reg [15:0] src_wide;
    (* anyseq *) reg [7:0] src_narrow;
    (* anyseq *) reg [11:0] src_odd;

    wire [8*16-1:0] out_wide;
    wire [4*8-1:0] out_narrow;
    wire [15:0] counter_wide;
    wire [15:0] counter_narrow;
    wire [2*12-1:0] out_odd;
    wire [15:0] counter_odd;

    localparam [31:0] SEED_WIDE = 32'hA5A5_5A5A;
    localparam [31:0] SEED_NARROW = 32'hDEAD_BEEF;

    sc_decorrelator #(
        .NUM_STREAMS(8),
        .STREAM_WIDTH(16),
        .SHIFT_SEED(SEED_WIDE)
    ) wide (
        .clk(clk),
        .rst_n(rst_n),
        .source_bits(src_wide),
        .decorrelated(out_wide),
        .formal_phase_counter(counter_wide)
    );

    sc_decorrelator #(
        .NUM_STREAMS(4),
        .STREAM_WIDTH(8),
        .SHIFT_SEED(SEED_NARROW)
    ) narrow (
        .clk(clk),
        .rst_n(rst_n),
        .source_bits(src_narrow),
        .decorrelated(out_narrow),
        .formal_phase_counter(counter_narrow)
    );

    sc_decorrelator #(
        .NUM_STREAMS(2),
        .STREAM_WIDTH(12),
        .SHIFT_SEED(SEED_WIDE)
    ) odd (
        .clk(clk),
        .rst_n(rst_n),
        .source_bits(src_odd),
        .decorrelated(out_odd),
        .formal_phase_counter(counter_odd)
    );

    // Rotation by s, written as the upper half of the doubled word shifted
    // left by s (not with the RTL's pair of shifts and modulo).
    function [15:0] rotl16(input [15:0] x, input [3:0] s);
        reg [31:0] doubled;
        begin
            doubled = {x, x} << s;
            rotl16 = doubled[31:16];
        end
    endfunction

    function [7:0] rotl8(input [7:0] x, input [2:0] s);
        reg [15:0] doubled;
        begin
            doubled = {x, x} << s;
            rotl8 = doubled[15:8];
        end
    endfunction

    function [11:0] rotl12(input [11:0] x, input [3:0] s);
        reg [23:0] doubled;
        begin
            doubled = {x, x} << s;
            rotl12 = doubled[23:12];
        end
    endfunction

    function [4:0] ones16(input [15:0] x);
        integer k;
        begin
            ones16 = 0;
            for (k = 0; k < 16; k = k + 1) ones16 = ones16 + x[k];
        end
    endfunction

    reg past_valid;
    reg past_valid2;
    initial past_valid = 1'b0;
    initial past_valid2 = 1'b0;
    reg prev_rst_n;
    reg [15:0] prev_wide;
    reg [7:0] prev_narrow;
    reg [11:0] prev_odd;

    always @(posedge clk) begin
        past_valid <= 1'b1;
        past_valid2 <= past_valid;
        prev_rst_n <= rst_n;
        prev_wide <= src_wide;
        prev_narrow <= src_narrow;
        prev_odd <= src_odd;
    end

    // Reset is asserted in the first clock and may come again at any time.
    always @(*) begin
        if (!past_valid) assume(!rst_n);
    end

    // Out of reset for this clock and the previous one.
    wire settled = past_valid && rst_n && prev_rst_n;

    // Shadow of the phase counter: zero in reset, one more every clock.
    reg [15:0] phase;
    reg [15:0] prev_phase;
    always @(posedge clk) begin
        phase <= rst_n ? phase + 16'd1 : 16'd0;
        prev_phase <= phase;
    end

    // P0: the RTL phase counter of both instances is this count. Asserting
    // it also makes the induction see the link between shadow and counter.
    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            assert(counter_wide == phase);
            assert(counter_narrow == phase);
            assert(counter_odd == phase);
        end
    end

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_wide
            localparam [3:0] NIBBLE = SEED_WIDE[4*i+3 -: 4];
            wire [15:0] word = out_wide[16*i +: 16];
            always @(posedge clk) begin
                if (settled && past_valid2) begin
                    assert(word == rotl16(prev_wide, i[3:0] ^ prev_phase[3:0] ^ NIBBLE));
                end
                if (!rst_n) assert(word == 16'd0);
            end
        end

        for (i = 0; i < 4; i = i + 1) begin : gen_narrow
            localparam [3:0] NIBBLE = SEED_NARROW[4*i+3 -: 4];
            wire [3:0] amount = i[3:0] ^ prev_phase[3:0] ^ NIBBLE;
            wire [7:0] word = out_narrow[8*i +: 8];
            always @(posedge clk) begin
                if (settled && past_valid2) begin
                    assert(word == rotl8(prev_narrow, amount[2:0]));
                end
                if (!rst_n) assert(word == 8'd0);
            end
        end

        for (i = 0; i < 2; i = i + 1) begin : gen_odd
            localparam [3:0] NIBBLE = SEED_WIDE[4*i+3 -: 4];
            wire [3:0] amount = i[3:0] ^ prev_phase[3:0] ^ NIBBLE;
            wire [3:0] reduced = (amount >= 4'd12) ? amount - 4'd12 : amount;
            wire [11:0] word = out_odd[12*i +: 12];
            always @(posedge clk) begin
                if (settled && past_valid2) begin
                    assert(word == rotl12(prev_odd, reduced));
                end
                if (!rst_n) assert(word == 12'd0);
            end
        end
    endgenerate

    // P3: distinct rotations of a one-hot word give distinct words.
    integer a;
    integer b;
    always @(posedge clk) begin
        if (settled && ones16(prev_wide) == 5'd1) begin
            for (a = 0; a < 8; a = a + 1)
                for (b = a + 1; b < 8; b = b + 1)
                    assert(out_wide[16*a +: 16] != out_wide[16*b +: 16]);
        end
    end

    // Stream 0 of the narrow instance (seed nibble 0xF) reaches a shift
    // amount of 8 or more with a non-zero source word: the case in which a
    // shift not reduced modulo the width emptied the word.
    always @(posedge clk) begin
        if (settled && past_valid2)
            cover(prev_narrow != 8'd0 && (4'hF ^ prev_phase[3:0]) >= 4'd8);
    end

endmodule
