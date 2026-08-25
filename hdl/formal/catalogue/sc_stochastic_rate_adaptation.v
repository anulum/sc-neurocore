// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

module sc_stochastic_rate_adaptation(
    input wire clk, input wire rst, input wire valid,
    input wire signed [31:0] next_adaptation_q,
    input wire [31:0] probability_q, input wire [31:0] uniform_q,
    output reg signed [31:0] adaptation_q, output reg spike
);
    always @(posedge clk) begin
        if (rst) begin adaptation_q <= 0; spike <= 0; end
        else if (valid) begin adaptation_q <= next_adaptation_q; spike <= uniform_q < probability_q; end
        else spike <= 0;
    end
endmodule
