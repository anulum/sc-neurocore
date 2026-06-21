`timescale 1ns/1ps
`default_nettype none
`include "timing_assertions.svh"

// Reference two-flop synchroniser device under proof: a clean destination-domain
// flop chain (no combinational logic between the flops). This is the shape a
// consumer's ingress primitive (e.g. MIF's `mif_aer_cdc_synchroniser.sv`) takes;
// the property template proves the chain is correct.
module two_flop_synchroniser (
    input  wire dst_clk,
    input  wire rst_n,
    input  wire async_in,
    output reg  meta_q,
    output reg  sync_out
);
    always @(posedge dst_clk or negedge rst_n) begin
        if (!rst_n) begin
            meta_q   <= 1'b0;
            sync_out <= 1'b0;
        end else begin
            meta_q   <= async_in;
            sync_out <= meta_q;
        end
    end
endmodule

// Formal harness: `async_src` is a free source-domain input (it may change on any
// destination edge), so the proof covers every source behaviour, including holds
// that exercise the data-integrity property and toggles that do not.
module example_cdc_two_flop_synchroniser (
    input wire clk,
    input wire rst_n,
    input wire async_src
);
    reg past_valid = 1'b0;

    wire meta_q;
    wire sync_out;

    always @(posedge clk) begin
        past_valid <= 1'b1;
        if (!past_valid) begin
            assume (!rst_n);
        end else begin
            assume (rst_n);
        end
    end

    two_flop_synchroniser dut (
        .dst_clk(clk),
        .rst_n(rst_n),
        .async_in(async_src),
        .meta_q(meta_q),
        .sync_out(sync_out)
    );

    `SC_ASSERT_CDC_TWO_FLOP(aer_ingress, clk, rst_n, async_src, meta_q, sync_out, 2)
endmodule

`default_nettype wire
