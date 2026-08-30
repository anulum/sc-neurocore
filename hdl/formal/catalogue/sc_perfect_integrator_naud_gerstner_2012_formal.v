// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal boundary contract for source perfect integrator

`default_nettype none

module sc_perfect_integrator_naud_gerstner_2012_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [15:0] I_t
);
    wire spike_out;
    wire signed [15:0] v_out;

    sc_perfect_integrator_naud_gerstner_2012 uut (
        .clk(clk), .rst_n(rst_n), .I_t(I_t),
        .spike_out(spike_out), .v_out(v_out)
    );

`ifdef FORMAL
    reg past_valid = 1'b0;
    always @(posedge clk)
        past_valid <= 1'b1;

    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end

    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            assert ($signed(v_out) >= -16'sd32768);
            assert ($signed(v_out) <= 16'sd32767);
            // A source event always exposes the configured reset state.
            if (spike_out)
                assert (v_out == 16'sd0);
        end
    end
`endif
endmodule
