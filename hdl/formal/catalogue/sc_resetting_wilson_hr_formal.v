// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_resetting_wilson_hr

`default_nettype none

// Formal wrapper for equation-compiler RTL of a retained SC project model.
// Properties use only public ports so default_nettype none stays clean.
module sc_resetting_wilson_hr_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [15:0] I_t
);

    wire spike_out;
    wire signed [15:0] v_out;
    wire signed [15:0] r_out;

    sc_resetting_wilson_hr uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .v_out(v_out),
        .r_out(r_out)
    );

`ifdef FORMAL
    // Reset values: while reset is held the spike flag is clear and every public
    // state port carries its encoded initial value, which need not be zero.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
            assert ($signed(v_out) == -16'sd179);
            assert ($signed(r_out) == 16'sd26);
        end
    end
`endif

endmodule
