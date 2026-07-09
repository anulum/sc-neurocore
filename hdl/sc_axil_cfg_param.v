// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parameterized AXI-Lite register file (Tier 3.3)
//
// Replaces hardcoded 3-input/7-neuron register file with parameterized
// N_INPUTS × N_NEURONS configuration. Each register is 16-bit Q8.8.

module sc_axil_cfg_param #(
    parameter N_INPUTS  = 4,
    parameter N_NEURONS = 8,
    parameter N_REGS    = N_INPUTS + N_NEURONS + 4,  // inputs + weights + control
    parameter ADDR_W    = $clog2(N_REGS) + 2          // byte-addressed
)(
    input  wire              clk,
    input  wire              rst,

    // AXI-Lite slave interface (simplified)
    input  wire [ADDR_W-1:0] s_axil_awaddr,
    input  wire              s_axil_awvalid,
    output reg               s_axil_awready,
    input  wire [31:0]       s_axil_wdata,
    input  wire              s_axil_wvalid,
    output reg               s_axil_wready,
    output reg  [1:0]        s_axil_bresp,
    output reg               s_axil_bvalid,
    input  wire              s_axil_bready,

    input  wire [ADDR_W-1:0] s_axil_araddr,
    input  wire              s_axil_arvalid,
    output reg               s_axil_arready,
    output reg  [31:0]       s_axil_rdata,
    output reg  [1:0]        s_axil_rresp,
    output reg               s_axil_rvalid,
    input  wire              s_axil_rready,

    // Register outputs to SC layer
    output wire [16*N_INPUTS-1:0]  input_probs,   // Q8.8 per input
    output wire [16*N_NEURONS-1:0] neuron_thresholds, // Q8.8 per neuron
    output wire                    layer_enable,
    output wire [15:0]             bitstream_length
);

    reg [31:0] regs [0:N_REGS-1];
    integer i;

    // Decode register addresses
    always @(posedge clk) begin
        if (rst) begin
            s_axil_awready <= 1'b0;
            s_axil_wready  <= 1'b0;
            s_axil_bvalid  <= 1'b0;
            s_axil_arready <= 1'b0;
            s_axil_rvalid  <= 1'b0;
            for (i = 0; i < N_REGS; i = i + 1)
                regs[i] <= 32'd0;
        end else begin
            // Write path
            s_axil_awready <= s_axil_awvalid && s_axil_wvalid && !s_axil_bvalid;
            s_axil_wready  <= s_axil_awvalid && s_axil_wvalid && !s_axil_bvalid;
            if (s_axil_awvalid && s_axil_wvalid && !s_axil_bvalid) begin
                regs[s_axil_awaddr[ADDR_W-1:2]] <= s_axil_wdata;
                s_axil_bvalid <= 1'b1;
                s_axil_bresp  <= 2'b00;
            end
            if (s_axil_bvalid && s_axil_bready)
                s_axil_bvalid <= 1'b0;

            // Read path
            s_axil_arready <= s_axil_arvalid && !s_axil_rvalid;
            if (s_axil_arvalid && !s_axil_rvalid) begin
                s_axil_rdata  <= regs[s_axil_araddr[ADDR_W-1:2]];
                s_axil_rvalid <= 1'b1;
                s_axil_rresp  <= 2'b00;
            end
            if (s_axil_rvalid && s_axil_rready)
                s_axil_rvalid <= 1'b0;
        end
    end

    // Map registers to outputs
    genvar g;
    generate
        for (g = 0; g < N_INPUTS; g = g + 1) begin : gen_inputs
            assign input_probs[16*g +: 16] = regs[g][15:0];
        end
        for (g = 0; g < N_NEURONS; g = g + 1) begin : gen_thresholds
            assign neuron_thresholds[16*g +: 16] = regs[N_INPUTS + g][15:0];
        end
    endgenerate

    assign layer_enable    = regs[N_INPUTS + N_NEURONS][0];
    assign bitstream_length = regs[N_INPUTS + N_NEURONS + 1][15:0];

endmodule
