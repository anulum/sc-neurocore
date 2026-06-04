// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Precision overflow trap latch RTL

`timescale 1ns / 1ps

module sc_precision_overflow_trap #(
    parameter integer TRAP_WIDTH = 1
)(
    input wire clk,
    input wire rst_n,
    input wire clear_trap,
    input wire [TRAP_WIDTH-1:0] overflow_in,
    output reg [TRAP_WIDTH-1:0] trap_vector,
    output wire [TRAP_WIDTH-1:0] trap_event_vector,
    output wire trap_event,
    output wire trap_latched
);

wire trap_accepting;

assign trap_accepting = rst_n & ~clear_trap;
assign trap_event_vector = trap_accepting ? overflow_in : {TRAP_WIDTH{1'b0}};
assign trap_event = |trap_event_vector;
assign trap_latched = |trap_vector;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        trap_vector <= {TRAP_WIDTH{1'b0}};
    end else if (clear_trap) begin
        trap_vector <= {TRAP_WIDTH{1'b0}};
    end else begin
        trap_vector <= trap_vector | trap_event_vector;
    end
end

`ifdef SC_NEUROCORE_ASSERTIONS
property p_no_silent_precision_overflow;
    @(posedge clk) disable iff (!rst_n || clear_trap)
        (|overflow_in) |-> (trap_event && (trap_event_vector == overflow_in));
endproperty

property p_sticky_precision_overflow;
    @(posedge clk) disable iff (!rst_n || clear_trap)
        (|overflow_in) |=> ((trap_vector & $past(overflow_in)) == $past(overflow_in));
endproperty

a_no_silent_precision_overflow: assert property (p_no_silent_precision_overflow);
a_sticky_precision_overflow: assert property (p_sticky_precision_overflow);
`endif

endmodule
