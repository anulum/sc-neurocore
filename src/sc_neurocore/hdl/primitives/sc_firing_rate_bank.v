// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Firing rate estimator bank

// hdl/sc_firing_rate_bank.v
//
// Firing rate estimator bank.
// Accumulates spikes over the run duration and scales the result.
//
// rate = (total_spikes * SCALE) >> SCALE_WIDTH
// where SCALE usually represents (1/T) * 2^SCALE_WIDTH.

`timescale 1ns / 1ps

module sc_firing_rate_bank #(
    parameter integer N_NEURONS = 7,
    parameter integer CNT_WIDTH = 16,
    parameter integer SCALE_WIDTH = 32
)(
    input wire                      clk,
    input wire                      rst_n,

    // Spike inputs from neurons
    input wire [N_NEURONS-1:0]      spikes,
    input wire                      step_valid, // High when spikes are valid

    // Run control
    input wire                      run_active, // While 1, accumulate
    input wire                      run_done,   // Pulse/high when done -> update output

    // Scale factor (fixed-point)
    input wire [SCALE_WIDTH-1:0]    SCALE_Q16,

    // Output rates (Q16.16 fixed-point)
    output reg [31:0]               rate_q16 [0:N_NEURONS-1]
);

    // Internal counters
    reg [CNT_WIDTH-1:0] accumulators [0:N_NEURONS-1];

    // Wide product to avoid overflow: CNT_WIDTH + SCALE_WIDTH bits
    reg [CNT_WIDTH + SCALE_WIDTH - 1:0] wide_product;

    integer i;

    // Accumulation logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                accumulators[i] <= {CNT_WIDTH{1'b0}};
                rate_q16[i]     <= 32'b0;
            end
        end else begin
            // Protocol: External controller drives run_active=1 during run,
            // then pulses run_done=1 (with run_active=0) to finalise.
            // Accumulators are cleared on run_done to prepare for next run.

            if (run_active) begin
                if (step_valid) begin
                    for (i = 0; i < N_NEURONS; i = i + 1) begin
                        if (spikes[i]) begin
                            accumulators[i] <= accumulators[i] + 1'b1;
                        end
                    end
                end
            end else if (run_done) begin
                // Finalise: rate_Q16 = count * SCALE_Q16 (scale is 1/T in Q16.16)
                for (i = 0; i < N_NEURONS; i = i + 1) begin
                    wide_product = accumulators[i] * SCALE_Q16;
                    rate_q16[i] <= wide_product[31:0];
                end
                // Clear accumulators for next run
                for (i = 0; i < N_NEURONS; i = i + 1) begin
                    accumulators[i] <= 0;
                end
            end else begin
                // Idle state
                // Keep accumulators at 0 if we cleared them.
            end
        end
    end

endmodule
