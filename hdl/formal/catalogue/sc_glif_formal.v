// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GLIF5 bounded reset-safety harness

`default_nettype none

module sc_glif_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] I_t
);

wire spike_out;
wire signed [63:0] v_out;
wire signed [63:0] theta_spike_out;
wire signed [63:0] i_asc1_out;
wire signed [63:0] i_asc2_out;
wire signed [63:0] theta_voltage_out;
wire [1:0] refractory_out;

sc_glif uut (
    .clk(clk),
    .rst_n(rst_n),
    .I_t(I_t),
    .spike_out(spike_out),
    .v_out(v_out),
    .theta_spike_out(theta_spike_out),
    .i_asc1_out(i_asc1_out),
    .i_asc2_out(i_asc2_out),
    .theta_voltage_out(theta_voltage_out),
    .refractory_out(refractory_out)
);

`ifdef FORMAL
reg past_valid = 1'b0;
reg reset_seen = 1'b0;
always @(posedge clk)
    past_valid <= 1'b1;
always @(posedge clk) begin
    if (!rst_n)
        reset_seen <= 1'b1;
end

always @(*) begin
    if (past_valid && reset_seen && !rst_n) begin
        assert (spike_out == 1'b0);
        assert (refractory_out == 2'd0);
    end
    if (past_valid && reset_seen)
        assert (refractory_out <= 2'd2);
    if (past_valid && reset_seen && spike_out)
        assert (v_out == -64'sd300647710720);
end
`endif

endmodule

`default_nettype wire
