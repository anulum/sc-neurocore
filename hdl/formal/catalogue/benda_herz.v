// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// Source Benda-Herz phase/adaptation datapath, signed Q16.16.
module benda_herz(
    input wire clk, input wire rst, input wire valid,
    input wire signed [31:0] rate_hz_q,
    output reg signed [31:0] adaptation_q,
    output reg [31:0] phase_q, output reg spike
);
    // Q16.16 coefficient words: round(0.1*2^16)=6554,
    // round(0.001*2^16)=66, round(0.0001*2^16)=7.
    wire signed [63:0] target_product = $signed(rate_hz_q) * 32'sd6554;
    wire signed [63:0] target = target_product >>> 16;
    wire signed [63:0] delta_product = (target - $signed(adaptation_q)) * 32'sd66;
    wire signed [63:0] delta_a = delta_product >>> 16;
    wire [63:0] phase_product = $unsigned(rate_hz_q) * 32'd7;
    wire [63:0] phase_delta = phase_product >> 16;
    wire [63:0] phase_candidate = $unsigned(phase_q) + phase_delta;
    always @(posedge clk) begin
        if (rst) begin adaptation_q <= 0; phase_q <= 0; spike <= 0; end
        else if (valid) begin
            adaptation_q <= $signed(adaptation_q) + $signed(delta_a[31:0]);
            if (phase_candidate >= 32'd65536) begin phase_q <= 0; spike <= 1; end
            else begin phase_q <= phase_candidate[31:0]; spike <= 0; end
        end else spike <= 0;
    end
endmodule
