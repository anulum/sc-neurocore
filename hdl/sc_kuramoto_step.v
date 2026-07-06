// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fixed-point Kuramoto single-integration-step phase core

//hdl/sc_kuramoto_step.v
//
// Combinational single Euler step of the Kuramoto model with an explicit
// pairwise coupling matrix K_nm:
//
//   theta_n_next = wrap( theta_n + dt * ( omega_n
//                        + sum_m K_nm * sin(theta_m - theta_n) ) )
//
// This implements the documented `sc.kuramoto_step` IR contract
// (dtheta/dt = omega + sum K sin(theta_m - theta_n)); any 1/N normalisation
// convention is folded into K by the caller. The datapath is noiseless and
// deterministic — the stochastic terms of the production Rust KuramotoSolver
// are out of scope for this synthesis primitive.
//
// Fixed-point format (fixed by the baked sine LUT): Q8.16 signed, 24-bit,
// 64-entry sine table. Phases are held in [0, PHASE_MODULUS) radians·2^16.
// The MAC defers all rounding: per-oscillator products K_nm·sin_nm are summed
// in a wide accumulator and shifted once, so the fixed-point result depends
// only on this module's arithmetic (mirrored bit-for-bit by the co-simulation
// oracle).
//
// Single-step contract: the caller must keep the per-step phase advance below
// 2*pi so the branch-free wrap functions need at most one modulus correction.

`timescale 1ns / 1ps

module sc_kuramoto_step #(
    parameter integer N_OSC = 2,
    parameter integer DATA_WIDTH = 24,
    parameter integer FRACTION = 16,
    parameter integer LUT_SIZE = 64,
    // Q(FRACTION) integration step dt.
    parameter signed [DATA_WIDTH-1:0] DT_FIXED = 655,               // 0.01 in Q8.16
    // Q(FRACTION) representation of the 2*pi / pi phase moduli.
    parameter signed [DATA_WIDTH-1:0] PHASE_MODULUS = 411775,       // round(2*pi * 2^16)
    parameter signed [DATA_WIDTH-1:0] HALF_PHASE_MODULUS = 205887   // round(pi   * 2^16)
)(
    input  wire signed [N_OSC*DATA_WIDTH-1:0]        phases_in,
    input  wire signed [N_OSC*DATA_WIDTH-1:0]        omega,
    input  wire signed [N_OSC*N_OSC*DATA_WIDTH-1:0]  coupling,
    output wire signed [N_OSC*DATA_WIDTH-1:0]        phases_out
);

    // Accumulator headroom: each K_nm·sin_nm product is up to 2*DATA_WIDTH bits;
    // eight extra bits let up to 256 oscillators be summed without overflow.
    localparam integer ACC_WIDTH = 2 * DATA_WIDTH + 8;

    // Wrap a candidate phase back into the canonical [0, PHASE_MODULUS) range.
    function automatic signed [DATA_WIDTH-1:0] wrap_phase;
        input signed [DATA_WIDTH-1:0] phase_value;
        reg signed [DATA_WIDTH-1:0] wrapped;
        begin
            wrapped = phase_value;
            if (wrapped >= PHASE_MODULUS) begin
                wrapped = wrapped - PHASE_MODULUS;
            end else if (wrapped < 0) begin
                wrapped = wrapped + PHASE_MODULUS;
            end
            wrap_phase = wrapped;
        end
    endfunction

    // Wrap a phase difference into the signed [-pi, pi] one-step range.
    function automatic signed [DATA_WIDTH-1:0] wrap_delta;
        input signed [DATA_WIDTH-1:0] delta_value;
        reg signed [DATA_WIDTH-1:0] wrapped;
        begin
            wrapped = delta_value;
            if (wrapped > HALF_PHASE_MODULUS) begin
                wrapped = wrapped - PHASE_MODULUS;
            end else if (wrapped < -HALF_PHASE_MODULUS) begin
                wrapped = wrapped + PHASE_MODULUS;
            end
            wrap_delta = wrapped;
        end
    endfunction

    // Q8.16 sine lookup indexed by the quantised phase; 64-entry table.
    function automatic signed [DATA_WIDTH-1:0] sin_lut;
        input signed [DATA_WIDTH-1:0] phase_value;
        reg signed [DATA_WIDTH-1:0] wrapped_phase;
        reg [5:0] lut_index;
        begin
            wrapped_phase = wrap_phase(phase_value);
            lut_index = (wrapped_phase * LUT_SIZE) / PHASE_MODULUS;
            case (lut_index)
                6'd0: sin_lut = 24'sd0;
                6'd1: sin_lut = 24'sd6424;
                6'd2: sin_lut = 24'sd12785;
                6'd3: sin_lut = 24'sd19024;
                6'd4: sin_lut = 24'sd25080;
                6'd5: sin_lut = 24'sd30893;
                6'd6: sin_lut = 24'sd36410;
                6'd7: sin_lut = 24'sd41576;
                6'd8: sin_lut = 24'sd46341;
                6'd9: sin_lut = 24'sd50660;
                6'd10: sin_lut = 24'sd54491;
                6'd11: sin_lut = 24'sd57798;
                6'd12: sin_lut = 24'sd60547;
                6'd13: sin_lut = 24'sd62714;
                6'd14: sin_lut = 24'sd64277;
                6'd15: sin_lut = 24'sd65220;
                6'd16: sin_lut = 24'sd65536;
                6'd17: sin_lut = 24'sd65220;
                6'd18: sin_lut = 24'sd64277;
                6'd19: sin_lut = 24'sd62714;
                6'd20: sin_lut = 24'sd60547;
                6'd21: sin_lut = 24'sd57798;
                6'd22: sin_lut = 24'sd54491;
                6'd23: sin_lut = 24'sd50660;
                6'd24: sin_lut = 24'sd46341;
                6'd25: sin_lut = 24'sd41576;
                6'd26: sin_lut = 24'sd36410;
                6'd27: sin_lut = 24'sd30893;
                6'd28: sin_lut = 24'sd25080;
                6'd29: sin_lut = 24'sd19024;
                6'd30: sin_lut = 24'sd12785;
                6'd31: sin_lut = 24'sd6424;
                6'd32: sin_lut = 24'sd0;
                6'd33: sin_lut = -24'sd6424;
                6'd34: sin_lut = -24'sd12785;
                6'd35: sin_lut = -24'sd19024;
                6'd36: sin_lut = -24'sd25080;
                6'd37: sin_lut = -24'sd30893;
                6'd38: sin_lut = -24'sd36410;
                6'd39: sin_lut = -24'sd41576;
                6'd40: sin_lut = -24'sd46341;
                6'd41: sin_lut = -24'sd50660;
                6'd42: sin_lut = -24'sd54491;
                6'd43: sin_lut = -24'sd57798;
                6'd44: sin_lut = -24'sd60547;
                6'd45: sin_lut = -24'sd62714;
                6'd46: sin_lut = -24'sd64277;
                6'd47: sin_lut = -24'sd65220;
                6'd48: sin_lut = -24'sd65536;
                6'd49: sin_lut = -24'sd65220;
                6'd50: sin_lut = -24'sd64277;
                6'd51: sin_lut = -24'sd62714;
                6'd52: sin_lut = -24'sd60547;
                6'd53: sin_lut = -24'sd57798;
                6'd54: sin_lut = -24'sd54491;
                6'd55: sin_lut = -24'sd50660;
                6'd56: sin_lut = -24'sd46341;
                6'd57: sin_lut = -24'sd41576;
                6'd58: sin_lut = -24'sd36410;
                6'd59: sin_lut = -24'sd30893;
                6'd60: sin_lut = -24'sd25080;
                6'd61: sin_lut = -24'sd19024;
                6'd62: sin_lut = -24'sd12785;
                6'd63: sin_lut = -24'sd6424;
                default: sin_lut = 24'sd0;
            endcase
        end
    endfunction

    genvar gi;
    generate
        for (gi = 0; gi < N_OSC; gi = gi + 1) begin : osc
            wire signed [DATA_WIDTH-1:0] omega_n = omega[gi*DATA_WIDTH +: DATA_WIDTH];
            wire signed [DATA_WIDTH-1:0] phase_n = phases_in[gi*DATA_WIDTH +: DATA_WIDTH];

            integer m;
            reg signed [DATA_WIDTH-1:0] phase_m;
            reg signed [DATA_WIDTH-1:0] diff_nm;
            reg signed [DATA_WIDTH-1:0] sin_nm;
            reg signed [DATA_WIDTH-1:0] k_nm;
            reg signed [ACC_WIDTH-1:0]  coupling_acc;

            // sum_m K_nm * sin(theta_m - theta_n) with rounding deferred to one shift.
            always @* begin
                coupling_acc = {ACC_WIDTH{1'b0}};
                for (m = 0; m < N_OSC; m = m + 1) begin
                    phase_m = phases_in[m*DATA_WIDTH +: DATA_WIDTH];
                    diff_nm = wrap_delta(phase_m - phase_n);
                    sin_nm = sin_lut(diff_nm);
                    k_nm = coupling[(gi*N_OSC + m)*DATA_WIDTH +: DATA_WIDTH];
                    coupling_acc = coupling_acc + (k_nm * sin_nm);
                end
            end

            wire signed [DATA_WIDTH-1:0] coupling_term = coupling_acc >>> FRACTION;
            wire signed [DATA_WIDTH-1:0] phase_velocity = omega_n + coupling_term;
            wire signed [2*DATA_WIDTH-1:0] delta_mult = phase_velocity * DT_FIXED;
            wire signed [DATA_WIDTH-1:0] phase_delta = delta_mult >>> FRACTION;
            wire signed [DATA_WIDTH-1:0] phase_next = wrap_phase(phase_n + phase_delta);

            assign phases_out[gi*DATA_WIDTH +: DATA_WIDTH] = phase_next;
        end
    endgenerate

endmodule
