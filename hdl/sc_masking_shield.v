// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Side-channel masking module for stochastic bitstreams
//
// First-order Boolean masking: every sensitive value is split into two
// shares (x = share_a ⊕ share_b) so that neither share alone reveals
// information about x via power/EM side-channels.
//
// For stochastic computing this is especially natural: the masking
// random stream is already available from the quasi-random source,
// and XOR-splitting is free in the SC domain.
//
// Architecture:
//   - On clock edge: share_a = data_in ⊕ mask_rng
//                    share_b = mask_rng
//   - Recombine:     share_a ⊕ share_b = data_in (verified by assertion)
//
// This module protects the data bus between the neuron core and
// the decision/output stage against first-order DPA/CPA attacks.

module sc_masking_shield #(
    parameter DATA_WIDTH = 16
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   enable,
    input  wire [DATA_WIDTH-1:0]  data_in,
    input  wire [DATA_WIDTH-1:0]  mask_rng,     // from quasi-random source
    output reg  [DATA_WIDTH-1:0]  share_a,
    output reg  [DATA_WIDTH-1:0]  share_b,
    output wire [DATA_WIDTH-1:0]  recombined,   // share_a ⊕ share_b (for verification)
    output reg                    shares_valid
);

    // Recombination — always available for downstream verification
    assign recombined = share_a ^ share_b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            share_a      <= {DATA_WIDTH{1'b0}};
            share_b      <= {DATA_WIDTH{1'b0}};
            shares_valid <= 1'b0;
        end else if (enable) begin
            share_a      <= data_in ^ mask_rng;
            share_b      <= mask_rng;
            shares_valid <= 1'b1;
        end else begin
            shares_valid <= 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Formal property: recombined must always equal data_in (when valid)
    // This is trivially true for XOR masking but serves as a canary
    // if someone modifies the masking logic incorrectly.
    // ------------------------------------------------------------------
`ifdef FORMAL
    reg [DATA_WIDTH-1:0] prev_data_in;
    reg                  prev_enable;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_data_in <= 0;
            prev_enable  <= 0;
        end else begin
            prev_data_in <= data_in;
            prev_enable  <= enable;
        end
    end

    always @(posedge clk) begin
        if (rst_n && prev_enable)
            assert(recombined == prev_data_in);
    end
`endif

endmodule
