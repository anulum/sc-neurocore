// SPDX-License-Identifier: AGPL-3.0-or-later
`default_nettype none
module energy_lif_formal(input wire clk,input wire rst_n,input wire signed[63:0]current_t);wire signed[63:0]v,e;wire event_out;energy_lif uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.v_out(v),.epsilon_out(e),.event_out(event_out));
`ifdef FORMAL
reg past_valid=0;initial assume(!rst_n);always@(posedge clk)begin assume(!rst_n);past_valid<=1;if(past_valid)assert(v==-64'sd261993005056&&e==64'sd1374389535&&!event_out);end
`endif
endmodule
`default_nettype wire
