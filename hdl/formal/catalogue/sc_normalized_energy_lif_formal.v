// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

`default_nettype none
module sc_normalized_energy_lif_formal(input wire clk,input wire rst_n,input wire signed[63:0]current_t);wire signed[63:0]v,e;wire event_out;sc_normalized_energy_lif uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.v_out(v),.epsilon_out(e),.event_out(event_out));
`ifdef FORMAL
reg past_valid=0;initial assume(!rst_n);always@(posedge clk)begin assume(current_t>=-64'sd214748364800&&current_t<=64'sd214748364800);if(!past_valid)assume(!rst_n);else assume(rst_n);past_valid<=1;if(past_valid&&$past(!rst_n))assert(v==-64'sd300647710720&&e==64'sd4294967296&&!event_out);if(past_valid&&rst_n)begin assert(v>=-64'sd858993459200&&v<=64'sd429496729600);assert(e>=0&&e<=64'sd4294967296);end end
`endif
endmodule
`default_nettype wire
