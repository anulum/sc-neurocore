// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_adex

`default_nettype none

// Formal wrapper for the generated Q8.8 AdEx recurrence.
module sc_adex_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [15:0] I_t
);

    wire spike_out;
    wire signed [15:0] v_out;
    wire signed [15:0] w_out;

    sc_adex uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .v_out(v_out),
        .w_out(w_out)
    );

`ifdef FORMAL
    reg past_valid = 1'b0;
    initial assume (!rst_n);
    always @(posedge clk) begin
        if (!past_valid)
            assume (!rst_n);
        else
            assume (rst_n);
        past_valid <= 1'b1;
    end

    // The generated asynchronous reset restores the maintained initial state.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
            assert ($signed(v_out) == -16'sd16640);
            assert ($signed(w_out) == 16'sd0);
        end
    end

    // A public event and the public reset state are committed atomically.
    always @(posedge clk) begin
        if (past_valid && rst_n && spike_out)
            assert ($signed(v_out) == -16'sd17408);
        cover (past_valid && rst_n && spike_out);
    end
`endif

endmodule

`default_nettype wire
