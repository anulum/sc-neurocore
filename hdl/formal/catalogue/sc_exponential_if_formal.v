// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_exponential_if

`default_nettype none

// Formal wrapper for equation-compiler RTL of a dual-axis perfect model.
// Properties use only public ports so default_nettype none stays clean.
module sc_exponential_if_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] I_t
);

    wire spike_out;
    wire signed [63:0] v_out;

    sc_exponential_if uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .v_out(v_out)
    );

`ifdef FORMAL
    reg f_past_valid = 1'b0;
    initial assume (!rst_n);
    always @(posedge clk)
        f_past_valid <= 1'b1;

    // Keep the proof inside the documented voltage-equivalent current domain.
    always @(*) begin
        assume (I_t >= -64'sd4294967296000000);
        assume (I_t <= 64'sd4294967296000000);
    end

    // Asynchronous reset clears the event output and restores resting voltage.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
            assert (v_out == 64'shffffffbf00000000);
        end
    end

    // A sampled reset remains visible on the next clocked observation.
    always @(posedge clk) begin
        if (f_past_valid && !$past(rst_n)) begin
            assert (spike_out == 1'b0);
            assert (v_out == 64'shffffffbf00000000);
        end

        // Every emitted compatibility-lane event commits the declared reset.
        if (f_past_valid && spike_out)
            assert (v_out == 64'shffffffbc00000000);

        cover (spike_out);
    end
`endif

endmodule
