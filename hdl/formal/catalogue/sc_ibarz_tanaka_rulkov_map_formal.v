// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_ibarz_tanaka_rulkov_map

`default_nettype none

// Formal wrapper for equation-compiler RTL of a dual-axis perfect model.
// Properties use only public ports so default_nettype none stays clean.
module sc_ibarz_tanaka_rulkov_map_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t
);

    wire spike_out;
    wire signed [31:0] v_out;
    wire signed [31:0] u_out;

    sc_ibarz_tanaka_rulkov_map uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .v_out(v_out),
        .u_out(u_out)
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

    // The generated asynchronous reset restores the source profile's exact
    // Q16.16 state and clears the event output.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
            assert ($signed(v_out) == -32'sd65536);
            assert ($signed(u_out) == -32'sd6554);
        end
    end

    // At a committed cycle, the public event output is exactly the previous
    // pre-state fourth-branch guard. Executing that branch commits v=-1.
    always @(posedge clk) begin
        if (past_valid && rst_n && $past(rst_n)) begin
            assert (
                spike_out
                == (
                    ($signed($past(v_out)) > 32'sd0)
                    && (
                        $signed($past(v_out))
                        >= ((32'sd65536 + $signed($past(I_t))) + $signed($past(u_out)))
                    )
                )
            );
            if (spike_out)
                assert ($signed(v_out) == -32'sd65536);
        end
    end
`endif

endmodule
