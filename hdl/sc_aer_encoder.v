// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Clocked priority AER encoder with coalesced pending bits

// SC-NeuroCore — AER spike encoder: neuron spike → address-event packet
//
// Converts spike outputs from a population of N neurons into
// Address-Event Representation (AER) packets. Idle inputs produce no events.
// One lowest-index pending bit is served per cycle; overlapping spikes on an
// already pending bit coalesce. Clearing the served bit takes precedence over
// a simultaneous spike on that bit. The timestamp records the service cycle.
//
// AER packet format: {valid, neuron_id, timestamp}
//   - valid: 1-bit flag
//   - neuron_id: log2(N_NEURONS)-bit address
//   - timestamp: TIMESTAMP_WIDTH-bit counter
//
// This replaces clock-driven dense readout with event-driven sparse
// output. For a 1000-neuron population firing at 10 Hz with 1 MHz clock,
// only 10 events/ms are generated vs 1000 reads/cycle in dense mode.

module sc_aer_encoder #(
    parameter N_NEURONS       = 128,
    parameter NEURON_ID_WIDTH = $clog2(N_NEURONS),
    parameter TIMESTAMP_WIDTH = 16
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire [N_NEURONS-1:0]         spike_vector,  // per-neuron spike flags
    input  wire [TIMESTAMP_WIDTH-1:0]   timestamp,     // global timestamp

    // AER output (single event per cycle, priority-encoded)
    output reg                          event_valid,
    output reg  [NEURON_ID_WIDTH-1:0]   event_neuron_id,
    output reg  [TIMESTAMP_WIDTH-1:0]   event_timestamp,

    // High while at least one previously captured spike remains pending
    output wire                         busy
);

    // Priority encoder: find lowest-index spiking neuron
    reg  [N_NEURONS-1:0] pending;
    wire [N_NEURONS-1:0] next_pending;
    wire has_pending;

    // Priority encoder: find lowest set bit
    integer i;
    reg [NEURON_ID_WIDTH-1:0] lowest_id;
    reg found;

    // Latch new spikes on each cycle, clear served ones
    assign next_pending = (pending | spike_vector) & ~(has_pending ? (1 << lowest_id) : 0);
    assign has_pending  = |pending;
    assign busy         = has_pending;

    always @(*) begin
        lowest_id = 0;
        found     = 0;
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            if (pending[i] && !found) begin
                lowest_id = i[NEURON_ID_WIDTH-1:0];
                found     = 1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending         <= 0;
            event_valid     <= 0;
            event_neuron_id <= 0;
            event_timestamp <= 0;
        end else begin
            pending <= next_pending;
            if (has_pending) begin
                event_valid     <= 1;
                event_neuron_id <= lowest_id;
                event_timestamp <= timestamp;
            end else begin
                event_valid <= 0;
            end
        end
    end

endmodule
