// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal Verification for SC Dense Layer Core

`default_nettype none

module sc_dense_layer_core_formal (
    input wire        clk,
    input wire        rst_n,
    input wire        start_pulse,
    input wire [31:0] stream_len,
    input wire [47:0] x_input_fp,   // 3 * 16
    input wire [47:0] weight_fp,    // 3 * 16
    input wire [15:0] y_min_fp,
    input wire [15:0] y_max_fp,
    input wire [15:0] cfg_leak,
    input wire [15:0] cfg_gain
);

    wire [15:0] I_t;
    wire [4:0]  spikes;
    wire        step_valid;
    wire        run_done;
    wire        running;

    sc_dense_layer_core #(
        .N_INPUTS(3),
        .N_NEURONS(5),
        .DATA_WIDTH(16)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start_pulse(start_pulse),
        .stream_len(stream_len),
        .x_input_fp(x_input_fp),
        .weight_fp(weight_fp),
        .y_min_fp(y_min_fp),
        .y_max_fp(y_max_fp),
        .cfg_leak(cfg_leak),
        .cfg_gain(cfg_gain),
        .I_t(I_t),
        .spikes(spikes),
        .step_valid(step_valid),
        .run_done(run_done),
        .running(running)
    );

`ifdef FORMAL
    reg past_valid = 0;
    always @(posedge clk)
        past_valid <= 1;

    // Constrain stream_len to a sane range (avoid 0 and overflow edge cases)
    always @* assume(stream_len >= 2 && stream_len <= 32'd1024);

    // 1. After reset, running=0 and run_done=0
    always @(posedge clk) begin
        if (past_valid && !rst_n) begin
            assert(running == 1'b0);
            assert(run_done == 1'b0);
        end
    end

    // 2. running and run_done are mutually exclusive
    always @(posedge clk) begin
        if (past_valid && rst_n)
            assert(!(running && run_done));
    end

    // 3. t_counter increments by 1 when running (and not at terminal count)
    always @(posedge clk) begin
        if (past_valid && rst_n && $past(running) && $past(rst_n)
            && $past(uut.t_counter) != $past(stream_len) - 1) begin
            assert(uut.t_counter == $past(uut.t_counter) + 1);
        end
    end

    // 4. run_done asserts exactly when t_counter reaches stream_len-1
    always @(posedge clk) begin
        if (past_valid && rst_n && $past(running) && $past(rst_n)
            && $past(uut.t_counter) == $past(stream_len) - 1) begin
            assert(run_done == 1'b1);
            assert(running == 1'b0);
        end
    end

    // 5. Cover: a complete run finishes
    always @(posedge clk) begin
        if (past_valid && rst_n)
            cover(run_done == 1'b1);
    end
`endif

endmodule
