// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_theta

`default_nettype none

// Formal wrapper for the Ermentrout-Kopell 1986 equation (2.5) Q16.16 Euler
// representative. Properties use only public ports.
module sc_theta_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t
);

    wire spike_out;
    wire signed [31:0] theta_out;

    sc_theta uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .theta_out(theta_out)
    );

`ifdef FORMAL
    reg past_valid = 1'b0;
    reg protocol_started = 1'b0;
    always @(posedge clk) begin
        past_valid <= 1'b1;

        // Initialise through reset, then hold the source receipt drive a=2.
        if (!protocol_started)
            assume (!rst_n);
        else
            assume (rst_n);
        assume ($signed(I_t) == 32'sd131072);
        protocol_started <= 1'b1;
    end

    // Reset hygiene: async reset clears the spike flag. Primary state may reset
    // to a non-zero rest / init (e.g. QIF v=-1, Izhikevich vr) — do not force 0.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
            assert (theta_out == 32'sd0);
        end
    end

    // Q16.16 phase remains on the schema's compact circle envelope. A source
    // passage through pi wraps to the negative side of the circle.
    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            assert ($signed(theta_out) >= -32'sd205888);
            assert ($signed(theta_out) <= 32'sd203817);
            if (spike_out)
                assert ($signed(theta_out) < 32'sd0);
        end
    end
`endif

endmodule
