// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Lapicque 1907 latch/reset formal harness

`default_nettype none

module sc_lapicque_1907_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] V_t
);

    wire spike_out;
    wire excited_out;
    wire signed [63:0] v_out;

    sc_lapicque_1907 uut (
        .clk(clk),
        .rst_n(rst_n),
        .V_t(V_t),
        .spike_out(spike_out),
        .excited_out(excited_out),
        .v_out(v_out)
    );

`ifdef FORMAL
    reg past_valid = 1'b0;
    always @(posedge clk)
        past_valid <= 1'b1;

    // Async reset clears all caller-visible experiment state.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
            assert (excited_out == 1'b0);
            assert (v_out == 64'sd0);
        end
    end

    // Once excitation is latched it remains latched, and no second event can
    // be emitted until the experiment is explicitly re-armed through reset.
    always @(posedge clk) begin
        if (past_valid && $past(rst_n) && rst_n) begin
            if ($past(excited_out)) begin
                assert (excited_out == 1'b1);
                assert (spike_out == 1'b0);
            end
            if (spike_out)
                assert (excited_out == 1'b1);
        end
    end

    always @(posedge clk)
        cover (rst_n && spike_out && excited_out);
`endif

endmodule
