// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

`default_nettype none
module sc_stochastic_rate_adaptation_formal(input wire clk,input wire rst,input wire signed [31:0] next_adaptation_q,input wire [31:0] probability_q,input wire [31:0] uniform_q);
wire signed [31:0] adaptation_q; wire spike;
sc_stochastic_rate_adaptation dut(clk,rst,1'b1,next_adaptation_q,probability_q,uniform_q,adaptation_q,spike);
`ifdef FORMAL
reg past_valid=0;
initial assume(rst);
always @(posedge clk) begin
    assume(rst); past_valid<=1;
    if(past_valid) assert(adaptation_q==0 && !spike);
end
`endif
endmodule
`default_nettype wire
