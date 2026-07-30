// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SC Compte 16-bin structured E-to-E ring representative

// This module computes one no-autapse circular E-to-E aggregate over a
// 16-bin coarse representation of SC-COMPTE-WM-NETWORK. Gates and weights are
// unsigned Q16.16. The accumulator is Q32.32 and the result truncates once to
// Q16.16 after all sixteen taps. The frozen weights are the rounded result of
// SCCompteWMNetworkSpec.connectivity_footprint("ee", 0, 16 uniform targets),
// retaining its exact-discrete-unit-mean normalization before the source==
// target term is excluded. A start accepted while idle produces done exactly
// sixteen processing cycles later. Loads and additional starts are ignored
// while busy.
//
// This is a synthesizable connectivity representative. It does not implement
// the 2,560-cell membrane/synapse state, Poisson drive, protocol, binary64 FFT,
// firing-time interpolation, physical timing, area, power, or silicon.

module sc_compte_wm_ring16 (
    input wire clk,
    input wire rst_n,
    input wire load_valid,
    input wire [3:0] load_index,
    input wire [31:0] load_gate_q1616,
    input wire start,
    input wire [3:0] target_bin,
    output reg busy,
    output reg done,
    output reg [31:0] aggregate_q1616
);
    reg [31:0] gates [0:15];
    reg [3:0] tap_index;
    reg [3:0] target_latched;
    reg [63:0] accumulator_q3232;
    integer reset_index;

    function automatic [31:0] weight_q1616(input [3:0] offset);
        begin
            case (offset)
                4'd0:  weight_q1616 = 32'd106168;
                4'd1:  weight_q1616 = 32'd80982;
                4'd2:  weight_q1616 = 32'd61755;
                4'd3:  weight_q1616 = 32'd59755;
                4'd4:  weight_q1616 = 32'd59714;
                4'd5:  weight_q1616 = 32'd59714;
                4'd6:  weight_q1616 = 32'd59714;
                4'd7:  weight_q1616 = 32'd59714;
                4'd8:  weight_q1616 = 32'd59714;
                4'd9:  weight_q1616 = 32'd59714;
                4'd10: weight_q1616 = 32'd59714;
                4'd11: weight_q1616 = 32'd59714;
                4'd12: weight_q1616 = 32'd59714;
                4'd13: weight_q1616 = 32'd59755;
                4'd14: weight_q1616 = 32'd61755;
                4'd15: weight_q1616 = 32'd80982;
                default: weight_q1616 = 32'd0;
            endcase
        end
    endfunction

    wire [3:0] circular_offset = target_latched - tap_index;
    wire [31:0] selected_weight_q1616 = weight_q1616(circular_offset);
    wire [63:0] selected_product_q3232 =
        (tap_index == target_latched)
            ? 64'd0
            : gates[tap_index] * selected_weight_q1616;
    wire [63:0] accumulated_next_q3232 =
        accumulator_q3232 + selected_product_q3232;

    always @(posedge clk) begin
        if (!rst_n) begin
            for (reset_index = 0; reset_index < 16; reset_index = reset_index + 1)
                gates[reset_index] <= 32'd0;
            tap_index <= 4'd0;
            target_latched <= 4'd0;
            accumulator_q3232 <= 64'd0;
            aggregate_q1616 <= 32'd0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (busy) begin
                if (tap_index == 4'd15) begin
                    aggregate_q1616 <= accumulated_next_q3232[47:16];
                    accumulator_q3232 <= 64'd0;
                    tap_index <= 4'd0;
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    accumulator_q3232 <= accumulated_next_q3232;
                    tap_index <= tap_index + 4'd1;
                end
            end else begin
                if (load_valid)
                    gates[load_index] <= load_gate_q1616;
                if (start) begin
                    target_latched <= target_bin;
                    accumulator_q3232 <= 64'd0;
                    tap_index <= 4'd0;
                    busy <= 1'b1;
                end
            end
        end
    end
endmodule
