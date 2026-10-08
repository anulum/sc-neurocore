// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Clocked AER encoder event, priority and reset contracts

`timescale 1ns/1ps
`default_nettype none

module tb_sc_aer_encoder;
    parameter N_NEURONS = 128;
    parameter TIMESTAMP_WIDTH = 16;
    localparam ID_WIDTH = $clog2(N_NEURONS);
    reg clk = 0;
    reg rst_n = 0;
    reg [N_NEURONS-1:0] spike_vector = 0;
    reg [TIMESTAMP_WIDTH-1:0] timestamp = 0;
    wire event_valid;
    wire [ID_WIDTH-1:0] event_neuron_id;
    wire [TIMESTAMP_WIDTH-1:0] event_timestamp;
    wire busy;
    reg [N_NEURONS-1:0] pulse;
    integer mask;
    integer neuron;
    integer cases = 0;
    integer clock_ticks = 0;
    integer burst_events;
    integer burst_service_cycles;
    integer burst_capture_cycles;

`ifdef AER_ENCODER_SYNTHESIZED
    // Generic synthesis resolves the production default parameters into fixed ports.
    sc_aer_encoder dut (
`else
    sc_aer_encoder #(
        .N_NEURONS(N_NEURONS), .TIMESTAMP_WIDTH(TIMESTAMP_WIDTH)
    ) dut (
`endif
        .clk(clk), .rst_n(rst_n), .spike_vector(spike_vector),
        .timestamp(timestamp), .event_valid(event_valid),
        .event_neuron_id(event_neuron_id), .event_timestamp(event_timestamp),
        .busy(busy)
    );

    always #5 clk = ~clk;
    always @(negedge clk) timestamp = timestamp + 1;
    always @(posedge clk) clock_ticks = clock_ticks + 1;

    task reset_encoder;
        begin
            @(negedge clk);
            rst_n = 0;
            spike_vector = 0;
            #1;
            if (event_valid !== 0 || busy !== 0 || event_neuron_id !== 0
                || event_timestamp !== 0)
                $fatal(1, "asynchronous reset did not clear outputs and pending state");
            @(negedge clk);
            rst_n = 1;
        end
    endtask

    task expect_event;
        input valid;
        input integer id;
        begin
            @(posedge clk);
            #1;
            if (event_valid !== valid)
                $fatal(1, "unexpected event-valid N=%0d expected=%0d actual=%0d",
                    N_NEURONS, valid, event_valid);
            if (valid && (event_neuron_id !== id[ID_WIDTH-1:0]
                || event_timestamp !== timestamp))
                $fatal(1, "wrong address or service timestamp expected=%0d actual=%0d",
                    id, event_neuron_id);
        end
    endtask

    task drain_pulse;
        input [N_NEURONS-1:0] vector;
        integer id;
        integer capture_clock;
        integer first_service_clock;
        integer last_service_clock;
        begin
            reset_encoder;
            spike_vector = vector;
            expect_event(0, 0);
            capture_clock = clock_ticks;
            first_service_clock = clock_ticks;
            last_service_clock = clock_ticks;
            burst_events = 0;
            if (busy !== (|vector)) $fatal(1, "captured pending mask not reflected by busy");
            @(negedge clk);
            spike_vector = 0;
            for (id = 0; id < N_NEURONS; id = id + 1) begin
                if (vector[id]) begin
                    expect_event(1, id);
                    if (burst_events == 0) first_service_clock = clock_ticks;
                    burst_events = burst_events + 1;
                    last_service_clock = clock_ticks;
                    @(negedge clk);
                end
            end
            expect_event(0, 0);
            if (busy !== 0) $fatal(1, "drained mask retained pending spikes");
            burst_service_cycles = (burst_events == 0) ? 0
                : last_service_clock - first_service_clock + 1;
            burst_capture_cycles = first_service_clock - capture_clock;
            cases = cases + 1;
        end
    endtask

    initial begin
`ifdef AER_ENCODER_SYNTHESIZED
        if (N_NEURONS != 128 || TIMESTAMP_WIDTH != 16)
            $fatal(1, "synthesized encoder profile requires the actual default parameters");
`endif
        // Exhaust every low-four-neuron mask, including idle and simultaneous arrivals.
        for (mask = 0; mask < 16; mask = mask + 1) drain_pulse(mask);
        // Check every address, including indices above an unsized integer's width.
        for (neuron = 0; neuron < N_NEURONS; neuron = neuron + 1) begin
            pulse = 0;
            pulse[neuron] = 1;
            drain_pulse(pulse);
        end
        // A full population drains exactly one pending address on every service cycle.
        drain_pulse({N_NEURONS{1'b1}});
        $display("MEASUREMENT N=%0d full_burst_events=%0d service_cycles=%0d capture_cycles=%0d",
            N_NEURONS, burst_events, burst_service_cycles, burst_capture_cycles);

        // Merge another address, coalesce an unserved bit, and clear a coincident served bit.
        reset_encoder;
        spike_vector = 10;
        expect_event(0, 0);
        @(negedge clk);
        spike_vector = 14;
        expect_event(1, 1);
        @(negedge clk);
        spike_vector = 0;
        expect_event(1, 2);
        expect_event(1, 3);
        expect_event(0, 0);
        if (busy !== 0) $fatal(1, "merged/coalesced pulse retained pending bits");
        cases = cases + 1;

        // Service timestamps follow the live counter even across its unsigned wrap.
        reset_encoder;
        #1;
        timestamp = {TIMESTAMP_WIDTH{1'b1}};
        spike_vector = 3;
        expect_event(0, 0);
        @(negedge clk);
        spike_vector = 0;
        expect_event(1, 0);
        expect_event(1, 1);
        expect_event(0, 0);
        cases = cases + 1;

        // Reset must discard queued addresses immediately, without later replay.
        reset_encoder;
        spike_vector = {N_NEURONS{1'b1}};
        expect_event(0, 0);
        @(negedge clk);
        spike_vector = 0;
        expect_event(1, 0);
        reset_encoder;
        expect_event(0, 0);
        expect_event(0, 0);
        cases = cases + 1;
        $display("PASS N=%0d cases=%0d", N_NEURONS, cases);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "AER encoder regression exceeded its finite clock budget");
    end
endmodule

`default_nettype wire
