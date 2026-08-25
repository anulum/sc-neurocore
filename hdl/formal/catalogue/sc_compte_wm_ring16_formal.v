// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — bounded safety harness for SC Compte ring16 representative

module sc_compte_wm_ring16_formal (
    input wire clk,
    input wire rst_n,
    input wire load_valid,
    input wire [3:0] load_index,
    input wire [31:0] load_gate_q1616,
    input wire start,
    input wire [3:0] target_bin
);
    wire busy;
    wire done;
    sc_compte_wm_ring16 uut (
        .clk(clk),
        .rst_n(rst_n),
        .load_valid(load_valid),
        .load_index(load_index),
        .load_gate_q1616(load_gate_q1616),
        .start(start),
        .target_bin(target_bin),
        .busy(busy),
        .done(done),
        .aggregate_q1616()
    );

    reg past_valid = 1'b0;
    reg pending = 1'b0;
    reg [5:0] age = 6'd0;

    initial assume(!rst_n);
    always @(posedge clk) begin
        if (!past_valid)
            assume(!rst_n);
        else
            assume(rst_n);
        past_valid <= 1'b1;

        if (past_valid && $past(!rst_n)) begin
            assert(!busy);
            assert(!done);
        end

        if (past_valid && rst_n) begin
            if (done)
                assert(!busy);
            if ($past(done))
                assert(!done);
        end

        if (!rst_n) begin
            pending <= 1'b0;
            age <= 6'd0;
        end else if (!pending && start && !busy) begin
            pending <= 1'b1;
            age <= 6'd0;
        end else if (pending && age < 6'd16) begin
            assert(busy);
            assert(!done);
            age <= age + 6'd1;
        end else if (pending) begin
            assert(age == 6'd16);
            assert(done);
            assert(!busy);
            pending <= 1'b0;
            age <= 6'd0;
        end

        cover(past_valid && rst_n && done);
    end
endmodule
