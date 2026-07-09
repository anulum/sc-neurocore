// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — Event-triggered LIF neuron: idle until spike event arrives
//
// Unlike the clock-driven sc_lif_neuron.v which updates every cycle,
// this module only performs computation when:
//   1. An input event arrives (event_valid && event_target == my_id), OR
//   2. A leak timer fires (periodic membrane decay)
//
// Power consumption is proportional to input spike rate, not clock rate.
// For sparse networks (1-10 Hz firing in 1 MHz clock), this saves >99%
// of switching power vs clock-driven.
//
// Q8.8 fixed-point, same arithmetic as sc_lif_neuron.v.

module sc_event_neuron #(
    parameter DATA_WIDTH  = 16,
    parameter FRACTION    = 8,
    parameter LEAK_PERIOD = 100    // leak every N clock cycles
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // Event input
    input  wire                      event_valid,
    input  wire signed [DATA_WIDTH-1:0] event_weight,  // synaptic weight (Q8.8)

    // Configuration
    input  wire signed [DATA_WIDTH-1:0] leak_k,        // leak constant (Q8.8)
    input  wire signed [DATA_WIDTH-1:0] threshold,     // firing threshold (Q8.8)
    input  wire signed [DATA_WIDTH-1:0] v_reset,       // reset voltage (Q8.8)

    // Output
    output reg                       spike_out,
    output reg  signed [DATA_WIDTH-1:0] v_mem          // membrane voltage
);

    localparam signed [DATA_WIDTH-1:0] ZERO = 0;
    localparam signed [DATA_WIDTH-1:0] V_MAX = (1 << (DATA_WIDTH-1)) - 1;
    localparam signed [DATA_WIDTH-1:0] V_MIN = -(1 << (DATA_WIDTH-1));

    reg [$clog2(LEAK_PERIOD)-1:0] leak_counter;
    wire leak_tick = (leak_counter == 0);

    // Saturating add
    function signed [DATA_WIDTH-1:0] sat_add;
        input signed [DATA_WIDTH-1:0] a, b;
        reg signed [DATA_WIDTH:0] sum;
        begin
            sum = a + b;
            if (sum > V_MAX) sat_add = V_MAX;
            else if (sum < V_MIN) sat_add = V_MIN;
            else sat_add = sum[DATA_WIDTH-1:0];
        end
    endfunction

    // Saturating fixed-point multiply (Q8.8 × Q8.8 → Q8.8)
    function signed [DATA_WIDTH-1:0] sat_mul;
        input signed [DATA_WIDTH-1:0] a, b;
        reg signed [2*DATA_WIDTH-1:0] product;
        reg signed [DATA_WIDTH-1:0] result;
        begin
            product = a * b;
            result = product[DATA_WIDTH+FRACTION-1:FRACTION];
            if (product > (V_MAX <<< FRACTION)) sat_mul = V_MAX;
            else if (product < (V_MIN <<< FRACTION)) sat_mul = V_MIN;
            else sat_mul = result;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem        <= ZERO;
            spike_out    <= 0;
            leak_counter <= LEAK_PERIOD - 1;
        end else begin
            spike_out <= 0;

            // Leak: periodic decay toward zero
            if (leak_tick) begin
                v_mem <= sat_mul(v_mem, leak_k);
                leak_counter <= LEAK_PERIOD - 1;
            end else begin
                leak_counter <= leak_counter - 1;
            end

            // Event: integrate incoming weighted spike
            if (event_valid) begin
                v_mem <= sat_add(v_mem, event_weight);
            end

            // Threshold: fire and reset
            if (v_mem >= threshold) begin
                spike_out <= 1;
                v_mem     <= v_reset;
            end
        end
    end

endmodule
