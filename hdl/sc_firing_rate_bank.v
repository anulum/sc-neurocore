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
    
    integer i;

    // Accumulation logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                accumulators[i] <= {CNT_WIDTH{1'b0}};
                rate_q16[i]     <= 32'b0;
            end
        end else begin
            // Reset on rising edge of run_active (start of new run)
            // We need a 1-cycle delay or edge detection for run_active start.
            // Simplified: if run_active is low, hold accumulators at 0? 
            // Better: reset when run_done was high previously?
            // Let's assume an external controller handles the pulse sequence.
            // If we see run_active transition 0->1, we reset.
            
            // For now, let's just accumulate while run_active is 1.
            // If run_active goes low, we stop.
            // To clear, we rely on the fact that rate_q16 is updated on run_done.
            // But we need to clear counters for the NEXT run.
            
            if (run_active) begin
                if (step_valid) begin
                    for (i = 0; i < N_NEURONS; i = i + 1) begin
                        if (spikes[i]) begin
                            accumulators[i] <= accumulators[i] + 1'b1;
                        end
                    end
                end
            end else if (run_done) begin
                // Update outputs
                for (i = 0; i < N_NEURONS; i = i + 1) begin
                    // rate = (count * scale) >> 16? 
                    // No, scale is Q16. count is integer. output is Q16.
                    // rate_Q16 = count * scale_Q16
                    // Wait, if scale is 1/T in Q16, then count * scale is rate in Q16.
                    // We need a multiplier.
                    // 16-bit count * 32-bit scale -> 48-bit result.
                    // We take the lower 32 bits? No, result fits in 32 bits if rate <= 65535.
                    
                    rate_q16[i] <= accumulators[i] * SCALE_Q16; 
                end
                // Clear accumulators for next run? 
                // Or maybe clear them when run_active starts again?
                // Let's clear them here for safety.
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
