// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Shared P-lane integer LIF datapath for the SHD cores
//
// Provenance: imported from the group's earlier design study as part of Golden Reference #0.
// Processes 128 neurons per hidden layer in G = 128 / P groups of P shared accumulators,
// keeping per-neuron membranes and lazy-leak step marks behind per-group clock gates.
`timescale 1ns / 1ps
module sc_lanes #(
    parameter integer P         = 8,
    parameter integer FRAC_BITS = 0,
    parameter integer THETA1    = 1,
    parameter integer THETA2    = 1,
    parameter integer V1_WIDTH  = 22,
    parameter integer V2_WIDTH  = 17,
    parameter integer LAZY      = 1,
    parameter integer GATING    = 1
)(
    input  wire              clk,
    input  wire              init,       // start of a sample: all membranes and marks to 0
    input  wire              layer,
    input  wire [6:0]        grp,
    input  wire              clr,
    input  wire              add,
    input  wire [3:0]        y,          // layer 1: count; layer 2: 1
    input  wire [P*8-1:0]    w,          // weights of the group's lanes, lane 0 in the low byte
    input  wire              leak,       // LAZY catch-up cycle
    input  wire              lif,        // update cycle
    input  wire [10:0]       t,          // logical step
    input  wire [10:0]       k_it,       // catch-up iteration index
    input  wire              in_l1,      // layer-1 spikes only for t < T + 15
    output wire [P-1:0]      spk,        // spikes of the group in the update cycle
    output wire              any_it,
    output wire [P-1:0]      it_v,       // lanes iterating (for counters)
    output wire [P-1:0]      upd_v       // lanes updated in the update cycle (for counters)
);
    localparam integer NH = 128, G = NH / P, A1W = 20;
    localparam integer GW = (G > 1) ? $clog2(G) : 1;
    localparam integer VW = (V1_WIDTH > V2_WIDTH) ? V1_WIDTH : V2_WIDTH;
    wire [GW-1:0] g = grp[GW-1:0];

    genvar l, q;
    generate
        for (l = 0; l < P; l = l + 1) begin : lane
            // ---- shared accumulator
            reg  signed [A1W-1:0] acc;
            wire signed [A1W-1:0] wv = $signed(w[8*l +: 8]);
            wire signed [A1W-1:0] prod = (y[0] ? wv : 0) + (y[1] ? (wv <<< 1) : 0)
                                       + (y[2] ? (wv <<< 2) : 0) + (y[3] ? (wv <<< 3) : 0);
            wire g_acc;
            sc_cg #(GATING) cg_acc (.clk(clk), .en(clr | add), .gclk(g_acc));
            always @(posedge g_acc) begin
                if (clr) acc <= 0;
                else if (add) acc <= acc + prod;
            end

            // ---- per-neuron state of this lane: neuron g * P + l, for each g, both layers
            wire signed [VW-1:0] v_sel;
            wire        [10:0]   last_sel;
            wire signed [VW-1:0] vs1 [0:G-1];
            wire signed [VW-1:0] vs2 [0:G-1];
            wire        [10:0]   ls1 [0:G-1];
            wire        [10:0]   ls2 [0:G-1];
            assign v_sel    = layer ? vs2[g] : vs1[g];
            assign last_sel = layer ? ls2[g] : ls1[g];

            // ---- LIF arithmetic on the selected neuron, at the layer's width
            wire signed [V1_WIDTH-1:0] v1 = v_sel[V1_WIDTH-1:0];
            wire signed [V2_WIDTH-1:0] v2 = v_sel[V2_WIDTH-1:0];
            wire signed [V1_WIDTH-1:0] a1 = acc;
            wire signed [V2_WIDTH-1:0] a2 = acc;
            wire signed [V1_WIDTH-1:0] lk1 = v1 - (v1 >>> 2);
            wire signed [V2_WIDTH-1:0] lk2 = v2 - (v2 >>> 2);
            wire signed [V1_WIDTH-1:0] u1 = lk1 + (a1 <<< FRAC_BITS);
            wire signed [V2_WIDTH-1:0] u2 = lk2 + (a2 <<< FRAC_BITS);
            wire tch  = (LAZY == 0) || (acc != 0);
            wire fixp = layer ? ((v2 >>> 2) == 0) : ((v1 >>> 2) == 0);
            assign it_v[l]  = (LAZY != 0) && leak && tch && (k_it < t - last_sel) && !fixp;
            assign upd_v[l] = lif && tch && (layer || in_l1);
            wire fire = layer ? (u2 >= THETA2) : (u1 >= THETA1);
            assign spk[l] = upd_v[l] && fire;
            wire signed [VW-1:0] v_new = it_v[l] ? (layer ? {{(VW-V2_WIDTH){lk2[V2_WIDTH-1]}}, lk2}
                                                          : {{(VW-V1_WIDTH){lk1[V1_WIDTH-1]}}, lk1})
                                       : fire ? {VW{1'b0}}
                                       : (layer ? {{(VW-V2_WIDTH){u2[V2_WIDTH-1]}}, u2}
                                                : {{(VW-V1_WIDTH){u1[V1_WIDTH-1]}}, u1});
            wire wr = it_v[l] | upd_v[l];

            for (q = 0; q < G; q = q + 1) begin : nrn
                reg signed [V1_WIDTH-1:0] sv1;
                reg signed [V2_WIDTH-1:0] sv2;
                reg        [10:0]         sl1, sl2;
                wire e1 = init | (wr && !layer && g == q);
                wire e2 = init | (wr && layer && g == q);
                wire gc1, gc2;
                sc_cg #(GATING) cg1 (.clk(clk), .en(e1), .gclk(gc1));
                sc_cg #(GATING) cg2 (.clk(clk), .en(e2), .gclk(gc2));
                always @(posedge gc1) begin
                    if (init) begin sv1 <= 0; sl1 <= 0; end
                    else if (wr && !layer && g == q) begin
                        sv1 <= v_new[V1_WIDTH-1:0];
                        if (upd_v[l]) sl1 <= t + 1;
                    end
                end
                always @(posedge gc2) begin
                    if (init) begin sv2 <= 0; sl2 <= 0; end
                    else if (wr && layer && g == q) begin
                        sv2 <= v_new[V2_WIDTH-1:0];
                        if (upd_v[l]) sl2 <= t + 1;
                    end
                end
                assign vs1[q] = {{(VW-V1_WIDTH){sv1[V1_WIDTH-1]}}, sv1};
                assign vs2[q] = {{(VW-V2_WIDTH){sv2[V2_WIDTH-1]}}, sv2};
                assign ls1[q] = sl1;
                assign ls2[q] = sl2;
            end
        end
    endgenerate
    assign any_it = |it_v;
endmodule
