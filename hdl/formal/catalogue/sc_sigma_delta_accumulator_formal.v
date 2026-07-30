// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — bounded retained bipolar accumulator safety harness
`default_nettype none
module sc_sigma_delta_accumulator_formal(input wire clk,input wire rst_n,input wire signed[63:0] current_t);
localparam signed[63:0] CURRENT_MIN=-64'sd17179869184;localparam signed[63:0] CURRENT_MAX=64'sd17179869184;localparam signed[63:0] LIMIT=64'sd4294967296000000;wire signed[63:0] sigma;wire signed[1:0] event_out;sc_sigma_delta_accumulator uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.sigma_out(sigma),.event_out(event_out));
`ifdef FORMAL
reg past_valid=1'b0;initial assume(!rst_n);always@(posedge clk)begin assume(current_t>=CURRENT_MIN&&current_t<=CURRENT_MAX);if(!past_valid)assume(!rst_n);else assume(rst_n);past_valid<=1'b1;if(past_valid&&$past(!rst_n))assert(sigma==0&&event_out==0);if(past_valid&&rst_n)begin assert(sigma>=-LIMIT&&sigma<=LIMIT);assert(event_out>=-1&&event_out<=1);end end
`endif
endmodule
`default_nettype wire
