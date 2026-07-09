// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

`timescale 1ns/1ps
`default_nettype none

module sc_latency_monitor #(
    parameter integer MAX_CYCLES = 1,
    parameter integer COUNTER_WIDTH = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start_event,
    input  wire                         end_event,
    output reg                          violation,
    output reg                          active,
    output reg  [COUNTER_WIDTH-1:0]     age
);
    localparam [COUNTER_WIDTH-1:0] MAX_COUNT = MAX_CYCLES;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            violation <= 1'b0;
            active <= 1'b0;
            age <= {COUNTER_WIDTH{1'b0}};
        end else if (violation) begin
            violation <= 1'b1;
            active <= active;
            age <= age;
        end else if (active && end_event) begin
            active <= 1'b0;
            age <= {COUNTER_WIDTH{1'b0}};
        end else if (!active && start_event) begin
            if (end_event) begin
                active <= 1'b0;
                age <= {COUNTER_WIDTH{1'b0}};
            end else begin
                active <= 1'b1;
                age <= {COUNTER_WIDTH{1'b0}};
            end
        end else if (active) begin
            if (age >= MAX_COUNT) begin
                violation <= 1'b1;
                active <= active;
                age <= age;
            end else begin
                age <= age + {{(COUNTER_WIDTH-1){1'b0}}, 1'b1};
            end
        end else begin
            active <= 1'b0;
            age <= {COUNTER_WIDTH{1'b0}};
        end
    end
endmodule

module sc_deadline_monitor #(
    parameter integer DEADLINE_CYCLES = 1,
    parameter integer COUNTER_WIDTH = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         deadline_start,
    input  wire                         completion_event,
    output wire                         violation,
    output wire                         active,
    output wire [COUNTER_WIDTH-1:0]     age
);
    sc_latency_monitor #(
        .MAX_CYCLES(DEADLINE_CYCLES),
        .COUNTER_WIDTH(COUNTER_WIDTH)
    ) deadline_monitor (
        .clk(clk),
        .rst_n(rst_n),
        .start_event(deadline_start),
        .end_event(completion_event),
        .violation(violation),
        .active(active),
        .age(age)
    );
endmodule

module sc_bounded_liveness_monitor #(
    parameter integer WINDOW_CYCLES = 1,
    parameter integer COUNTER_WIDTH = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         request_event,
    input  wire                         witness_event,
    output wire                         violation,
    output wire                         active,
    output wire [COUNTER_WIDTH-1:0]     age
);
    sc_latency_monitor #(
        .MAX_CYCLES(WINDOW_CYCLES),
        .COUNTER_WIDTH(COUNTER_WIDTH)
    ) liveness_monitor (
        .clk(clk),
        .rst_n(rst_n),
        .start_event(request_event),
        .end_event(witness_event),
        .violation(violation),
        .active(active),
        .age(age)
    );
endmodule

// Two-flop clock-domain-crossing synchroniser property monitor.
//
// Reusable framework template (NEU-C.2) a consumer binds over the
// destination-domain synchroniser flops it owns (e.g. MIF's AER-ingress
// `mif_aer_cdc_synchroniser.sv`; META_Q = first flop, SYNC_OUT = last flop). The
// monitor builds an internal SYNC_DEPTH-deep reference delay of `async_in` and,
// once warmed up, latches `violation` on either:
//   1. Depth/data: `sync_out` is not `async_in` delayed by exactly `SYNC_DEPTH`
//      flops. This pins both the synchroniser depth (a 1-flop chain fails) and the
//      absence of combinational contamination on the data path.
//   2. Structural: `sync_out` is not a pure registered copy of `meta_q`
//      (`sync_out[t] != meta_q[t-1]`), i.e. a combinational/glitch path escapes
//      past the last flop. A pure flop output is also stable for the whole cycle,
//      meeting the "stable for >=1 destination cycle" requirement.
// `crossing_seen` latches a `sync_out` transition for a liveness cover, so a proof
// is not vacuous. This is the RTL-level structural check tractable in the
// open-source single-clock BMC flow; physical metastability resolution time
// remains a post-route/silicon fact, not a framework claim, so the monitor never
// asserts that the first flop is metastability-free.
module sc_cdc_two_flop_monitor #(
    parameter integer SYNC_DEPTH = 2,
    parameter integer COUNTER_WIDTH = 8
) (
    input  wire                         dst_clk,
    input  wire                         rst_n,
    input  wire                         async_in,
    input  wire                         meta_q,
    input  wire                         sync_out,
    output reg                          violation,
    output reg                          crossing_seen,
    output reg  [COUNTER_WIDTH-1:0]     warmup_age
);
    localparam [COUNTER_WIDTH-1:0] DEPTH_TARGET = SYNC_DEPTH;

    // async_d[k] holds `async_in` delayed by (k + 1) destination cycles, so
    // async_d[SYNC_DEPTH-1] is the source value SYNC_DEPTH cycles ago and
    // async_d[SYNC_DEPTH-2] is it SYNC_DEPTH-1 cycles ago (the expected meta_q).
    reg [SYNC_DEPTH-1:0] async_d;
    reg                  meta_q_prev;
    reg                  sync_out_prev;

    always @(posedge dst_clk or negedge rst_n) begin
        if (!rst_n) begin
            violation     <= 1'b0;
            crossing_seen <= 1'b0;
            warmup_age    <= {COUNTER_WIDTH{1'b0}};
            async_d       <= {SYNC_DEPTH{1'b0}};
            meta_q_prev   <= 1'b0;
            sync_out_prev <= 1'b0;
        end else begin
            // Liveness witness: a transition reached the synchroniser output.
            if ((warmup_age >= DEPTH_TARGET) && (sync_out != sync_out_prev))
                crossing_seen <= 1'b1;

            // Latched property checks, gated until the reference delay is filled.
            if (!violation && (warmup_age >= DEPTH_TARGET)) begin
                if (sync_out != async_d[SYNC_DEPTH-1])
                    violation <= 1'b1;
                else if (sync_out != meta_q_prev)
                    violation <= 1'b1;
            end

            async_d       <= {async_d[SYNC_DEPTH-2:0], async_in};
            meta_q_prev   <= meta_q;
            sync_out_prev <= sync_out;
            if (warmup_age < DEPTH_TARGET)
                warmup_age <= warmup_age + {{(COUNTER_WIDTH-1){1'b0}}, 1'b1};
        end
    end
endmodule

`default_nettype wire
