// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fixed-point degree-normalised graph aggregation core

//hdl/sc_graph_forward.v
//
// Combinational degree-normalised neighbourhood aggregation — the message-passing
// stage of a graph layer:
//
//   agg[i][f] = ( sum_j adjacency[i][j] * features[j][f] ) / degree[i]
//   degree[i] =   sum_j adjacency[i][j]
//
// This implements the aggregation half of the reference StochasticGraphLayer
// rate-mode forward pass (engine/src/graph.rs): the `sc.graph_forward` IR op
// carries only the `features` and `adjacency` operands, so it lowers exactly the
// graph-structural aggregation. The subsequent learnable weight transform and
// tanh activation depend on the layer's weight matrix (not an IR operand) and
// compose downstream as a dense layer.
//
// Fixed-point format: Q(FRACTION) signed, DATA_WIDTH-bit, two's-complement.
// Features are laid out row-major node-major: element (j, f) occupies word
// (j*N_FEATURES + f); adjacency element (i, j) occupies word (i*N_NODES + j);
// the aggregate output (i, f) occupies word (i*N_FEATURES + f); word 0 is at the
// bus LSB.
//
// Rounding is deferred: per-output products adjacency*features are summed in a
// wide Q(2*FRACTION) accumulator, then a single signed division by the Q(FRACTION)
// degree yields the Q(FRACTION) aggregate (num_Q2F / deg_QF == agg_QF). A zero
// row-degree leaves the sum — which is necessarily zero when every adjacency entry
// is zero — un-normalised, matching the reference `if degree != 0` guard. The
// result therefore depends only on this module's arithmetic and is mirrored
// bit-for-bit by the co-simulation oracle. Verilog signed division truncates
// toward zero.

`timescale 1ns / 1ps

module sc_graph_forward #(
    parameter integer N_NODES = 2,
    parameter integer N_FEATURES = 2,
    parameter integer DATA_WIDTH = 24,
    parameter integer FRACTION = 16
)(
    input  wire signed [N_NODES*N_FEATURES*DATA_WIDTH-1:0] features,
    input  wire signed [N_NODES*N_NODES*DATA_WIDTH-1:0]    adjacency,
    output wire signed [N_NODES*N_FEATURES*DATA_WIDTH-1:0] agg
);

    // Q(2*FRACTION) product accumulator with eight extra bits: up to 256
    // neighbour products can be summed without overflow.
    localparam integer ACC_WIDTH = 2 * DATA_WIDTH + 8;
    // Q(FRACTION) degree row-sum with the same 256-neighbour headroom.
    localparam integer DEG_WIDTH = DATA_WIDTH + 8;

    genvar gi, gf;
    generate
        for (gi = 0; gi < N_NODES; gi = gi + 1) begin : row
            integer jd;
            reg signed [DEG_WIDTH-1:0] degree_acc;

            // degree[i] = sum_j adjacency[i][j]
            always @* begin
                degree_acc = {DEG_WIDTH{1'b0}};
                for (jd = 0; jd < N_NODES; jd = jd + 1) begin
                    degree_acc = degree_acc
                        + $signed(adjacency[(gi*N_NODES + jd)*DATA_WIDTH +: DATA_WIDTH]);
                end
            end

            for (gf = 0; gf < N_FEATURES; gf = gf + 1) begin : feat
                integer jm;
                reg signed [DATA_WIDTH-1:0] a_ij;
                reg signed [DATA_WIDTH-1:0] x_jf;
                reg signed [ACC_WIDTH-1:0]  num_acc;
                reg signed [ACC_WIDTH-1:0]  quotient;

                // num[i][f] = sum_j adjacency[i][j] * features[j][f]
                always @* begin
                    num_acc = {ACC_WIDTH{1'b0}};
                    for (jm = 0; jm < N_NODES; jm = jm + 1) begin
                        a_ij = adjacency[(gi*N_NODES + jm)*DATA_WIDTH +: DATA_WIDTH];
                        x_jf = features[(jm*N_FEATURES + gf)*DATA_WIDTH +: DATA_WIDTH];
                        num_acc = num_acc + (a_ij * x_jf);
                    end
                end

                // Degree-normalise: Q(2F) numerator / Q(F) degree = Q(F) aggregate.
                always @* begin
                    if (degree_acc == {DEG_WIDTH{1'b0}}) begin
                        quotient = num_acc >>> FRACTION;
                    end else begin
                        quotient = num_acc / degree_acc;
                    end
                end

                assign agg[(gi*N_FEATURES + gf)*DATA_WIDTH +: DATA_WIDTH]
                    = quotient[DATA_WIDTH-1:0];
            end
        end
    endgenerate

endmodule
