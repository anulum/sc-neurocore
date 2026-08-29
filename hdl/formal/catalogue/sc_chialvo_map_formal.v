// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_chialvo_map

`default_nettype none

// Formal wrapper for equation-compiler RTL of the Q8.8 Chialvo map.
module sc_chialvo_map_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [15:0] I_t
);

    wire spike_out;
    wire signed [15:0] x_out;
    wire signed [15:0] y_out;

    sc_chialvo_map uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .x_out(x_out),
        .y_out(y_out)
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

    // The generated asynchronous reset clears both public states and the
    // maintained event output.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
            assert ($signed(x_out) == 16'sd0);
            assert ($signed(y_out) == 16'sd0);
        end
    end

    // At every committed cycle the event output is exactly the maintained
    // upward crossing of the public fast-variable state at Q8.8 threshold 1.
    always @(posedge clk) begin
        if (past_valid && rst_n && $past(rst_n)) begin
            assert (
                spike_out
                == (
                    ($signed($past(x_out)) < 16'sd256)
                    && ($signed(x_out) >= 16'sd256)
                )
            );
        end
        cover (past_valid && rst_n && spike_out);
    end
`endif

endmodule

`default_nettype wire
