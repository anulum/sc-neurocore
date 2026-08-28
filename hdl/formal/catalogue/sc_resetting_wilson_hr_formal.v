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
    reg past_valid = 1'b0;
    always @(posedge clk)
        past_valid <= 1'b1;

    // Reset hygiene: async reset clears the spike flag. Primary state may reset
    // to a non-zero rest / init (e.g. QIF v=-1, Izhikevich vr) — do not force 0.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
        end
    end

    // Saturation contract on the primary membrane / phase / current state.
    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            assert ($signed(v_out) >= -16'sd32768);
            assert ($signed(v_out) <= 16'sd32767);
        end
    end
`endif

endmodule
