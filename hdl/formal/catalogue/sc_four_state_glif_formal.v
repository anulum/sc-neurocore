// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_four_state_glif

`default_nettype none

// Formal wrapper for equation-compiler RTL of a retained SC project model.
// Properties use only public ports so default_nettype none stays clean.
module sc_four_state_glif_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t
);

    wire spike_out;
    wire signed [31:0] v_out;
    wire signed [31:0] theta_out;
    wire signed [31:0] i_asc1_out;
    wire signed [31:0] i_asc2_out;

    sc_four_state_glif uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .v_out(v_out),
        .theta_out(theta_out),
        .i_asc1_out(i_asc1_out),
        .i_asc2_out(i_asc2_out)
    );

`ifdef FORMAL
    // Reset values: while reset is held the spike flag is clear and every public
    // state port carries its encoded initial value, which need not be zero.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
            assert ($signed(v_out) == -32'sd4587520);
            assert ($signed(theta_out) == -32'sd3276800);
            assert ($signed(i_asc1_out) == 32'sd0);
            assert ($signed(i_asc2_out) == 32'sd0);
        end
    end
`endif

endmodule
