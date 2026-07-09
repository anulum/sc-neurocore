// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — Switching activity comparison: clock-driven vs event-driven neuron
//
// Measures register toggle counts over N clock cycles with sparse input events.
// Toggle count is proportional to dynamic power (P_dyn ~ C * V^2 * f * alpha,
// where alpha = toggle rate).
//
// Usage: iverilog -o tb_power tb_power_comparison.v sc_lif_neuron.v sc_event_neuron.v && vvp tb_power

`timescale 1ns / 1ps

module tb_power_comparison;
    parameter DATA_WIDTH = 16;
    parameter FRACTION = 8;
    parameter N_CYCLES = 10000;
    parameter SPIKE_INTERVAL = 100; // 1 spike every 100 cycles = 1% activity

    reg clk, rst_n;
    reg signed [DATA_WIDTH-1:0] I_clock;    // clock-driven input current
    reg event_valid;                         // event-driven trigger
    reg signed [DATA_WIDTH-1:0] event_weight;

    wire spike_clock, spike_event;
    wire signed [DATA_WIDTH-1:0] v_clock, v_event;

    // Fixed config
    wire signed [DATA_WIDTH-1:0] leak_k = 16'sd26;   // ~0.1 in Q8.8
    wire signed [DATA_WIDTH-1:0] gain_k = 16'sd256;  // 1.0 in Q8.8
    wire signed [DATA_WIDTH-1:0] threshold = 16'sd256; // 1.0
    wire signed [DATA_WIDTH-1:0] v_reset = 16'sd0;

    // Clock-driven neuron (updates every cycle)
    sc_lif_neuron #(.DATA_WIDTH(DATA_WIDTH), .FRACTION(FRACTION)) u_clock (
        .clk(clk), .rst_n(rst_n),
        .leak_k(leak_k), .gain_k(gain_k),
        .I_t(I_clock), .noise_in(16'sd0),
        .spike_out(spike_clock), .v_out(v_clock)
    );

    // Event-driven neuron (updates only on events + leak ticks)
    sc_event_neuron #(
        .DATA_WIDTH(DATA_WIDTH), .FRACTION(FRACTION), .LEAK_PERIOD(SPIKE_INTERVAL)
    ) u_event (
        .clk(clk), .rst_n(rst_n),
        .event_valid(event_valid), .event_weight(event_weight),
        .leak_k(leak_k), .threshold(threshold), .v_reset(v_reset),
        .spike_out(spike_event), .v_mem(v_event)
    );

    // Toggle counters
    reg signed [DATA_WIDTH-1:0] prev_v_clock, prev_v_event;
    integer toggle_clock, toggle_event;
    integer cycle;
    integer bit_idx;

    // Count bit toggles between consecutive values
    function integer count_toggles;
        input [DATA_WIDTH-1:0] prev_val;
        input [DATA_WIDTH-1:0] curr_val;
        integer t, b;
        begin
            t = 0;
            for (b = 0; b < DATA_WIDTH; b = b + 1)
                t = t + (prev_val[b] ^ curr_val[b]);
            count_toggles = t;
        end
    endfunction

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0;
        I_clock = 0;
        event_valid = 0;
        event_weight = 16'sd128; // 0.5 in Q8.8
        toggle_clock = 0;
        toggle_event = 0;
        prev_v_clock = 0;
        prev_v_event = 0;

        #20 rst_n = 1;

        for (cycle = 0; cycle < N_CYCLES; cycle = cycle + 1) begin
            @(posedge clk);

            // Sparse input: 1 spike every SPIKE_INTERVAL cycles
            if (cycle % SPIKE_INTERVAL == 0) begin
                I_clock = 16'sd128;      // inject current for clock-driven
                event_valid = 1;          // trigger event-driven
            end else begin
                I_clock = 16'sd0;
                event_valid = 0;
            end

            // Count toggles on membrane potential registers
            @(negedge clk);
            toggle_clock = toggle_clock + count_toggles(prev_v_clock, v_clock);
            toggle_event = toggle_event + count_toggles(prev_v_event, v_event);
            prev_v_clock = v_clock;
            prev_v_event = v_event;
        end

        $display("=== SC-NeuroCore Power Comparison ===");
        $display("Cycles:         %0d", N_CYCLES);
        $display("Spike interval: %0d (%.1f%% activity)", SPIKE_INTERVAL, 100.0 / SPIKE_INTERVAL);
        $display("");
        $display("Clock-driven (sc_lif_neuron):");
        $display("  v_mem toggles: %0d", toggle_clock);
        $display("  Toggles/cycle: %.2f", 1.0 * toggle_clock / N_CYCLES);
        $display("");
        $display("Event-driven (sc_event_neuron):");
        $display("  v_mem toggles: %0d", toggle_event);
        $display("  Toggles/cycle: %.2f", 1.0 * toggle_event / N_CYCLES);
        $display("");
        if (toggle_event > 0)
            $display("Toggle reduction: %.1fx", 1.0 * toggle_clock / toggle_event);
        else
            $display("Toggle reduction: INF (event neuron never toggled)");
        $display("Power savings estimate: %.0f%%",
            100.0 * (1.0 - 1.0 * toggle_event / (toggle_clock > 0 ? toggle_clock : 1)));
        $finish;
    end

endmodule
