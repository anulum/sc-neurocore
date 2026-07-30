// SPDX-License-Identifier: AGPL-3.0-or-later
`default_nettype none
module sc_triangular_mckean_formal(input wire clk,input wire rst_n,input wire signed [63:0] current_t);
wire signed [63:0] v,w;wire event_out;
sc_triangular_mckean uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.v_out(v),.w_out(w),.event_out(event_out));
`ifdef FORMAL
reg past_valid=0;
initial assume(!rst_n);
always @(posedge clk) begin
    assume(!rst_n);past_valid<=1;
    if(past_valid) assert(v==0&&w==0&&!event_out);
end
`endif
endmodule
`default_nettype wire
