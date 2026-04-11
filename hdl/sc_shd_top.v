// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Top-level SHD inference network (Masquelier model)
//
// End-to-end Q8.8 fixed-point implementation of the Masquelier SHD speech
// recognition SNN, hardcoded to the network geometry of the released
// dcls_max checkpoint:
//
//      Input(140) -> AxonalDelay -> Dense(140->128) -> Vmin_LIF
//                 -> AxonalDelay -> Dense(128->128) -> Vmin_LIF
//                 -> Dense(128->20) -> Sum_v -> argmax
//
// Each `dcls_module` in the original PyTorch model uses asymmetric padding
// (left=30, right=15), so the OUTPUT time axis grows by 15 cycles past
// each axonal delay layer (T -> T+15 -> T+30). To match this exactly we
// run the network for `T_orig + 30` clock cycles per inference and mask
// the layer-1 spike output to zero from cycle `T_orig + 15` onwards
// (the equivalent of the dcls_l2 right-padding zeros that PyTorch sees).
//
// The full datapath is combinational from spike_in -> dense3_out_comb
// within a single clock cycle: every layer exposes a `_comb` port that
// taps the same combinational expression as the registered output of the
// stand-alone modules. The only sequential elements are the axonal-delay
// circular buffers, the Vmin_LIF membrane voltage registers, the cycle
// counter and the per-class output accumulator. This makes one clock
// equal to one full network step, which mirrors
// tools/shd_q88_reference.py::run_inference_q88 cycle-by-cycle and lets
// the cosim assert bit-true equality of `output_v_sum` between the two.
//
// Weights, delays and scales come from the artifact files emitted by
// tools/extract_shd_weights.py. The cosim Python harness writes them
// into a temporary directory and runs `vvp` from there so the relative
// paths below resolve. Per-tensor scales are passed as runtime input
// ports so a single compiled testbench can iterate over multiple
// checkpoints without re-elaboration.
//
// Verified by:
//   hdl/tb_sc_shd_top.v
//   tools/cosim_shd_top_verilog.py

`timescale 1ns / 1ps

module sc_shd_top (
    input  wire                   clk,
    input  wire                   rst_n,
    // Per-inference control
    input  wire                   start,             // pulse to begin a sample
    input  wire [15:0]            t_orig,            // input length, max ~1000
    // Streaming spike input — driven by the testbench, one 140-bit vector
    // per cycle, valid while `running == 1`. Bits past `t_orig` should be 0
    // but the top module also gates internally for safety.
    input  wire [139:0]           spike_in,
    // Per-tensor Q16.16 scales for the three dense layers
    input  wire signed [31:0]     scale_l1_q16_16,
    input  wire signed [31:0]     scale_l2_q16_16,
    input  wire signed [31:0]     scale_l3_q16_16,
    // Status / outputs
    output reg                    running,
    output reg                    done,
    output reg signed [20*32-1:0] output_v_sum_packed
);

    // ------------------------------------------------------------------
    // Constants — SHD network shape
    // ------------------------------------------------------------------
    localparam integer N_INPUT  = 140;
    localparam integer N_HIDDEN = 128;
    localparam integer N_OUTPUT = 20;
    localparam integer MAX_DELAY = 31;
    localparam integer DELAY_PTR_WIDTH = 5;
    localparam integer DELAY_HALF = (MAX_DELAY - 1) / 2;  // 15

    // ------------------------------------------------------------------
    // Cycle counter and gating signals
    // ------------------------------------------------------------------
    // `cycle` represents the iteration index of the iter being processed
    // AT the current posedge — equivalently, the value of Python's loop
    // counter t inside run_inference_q88. The reset/start branch sets it
    // so that this invariant holds from the very first posedge onwards
    // (start latches the iter-0 contribution and immediately increments
    // cycle to 1 = the next iter). All combinational gating signals use
    // `cycle` directly to mirror Python's `if t < ...` checks.
    reg  [15:0] cycle;
    wire        in_input_window  = (cycle < t_orig);                       // t < T
    wire        in_l1_window     = (cycle < (t_orig + DELAY_HALF));        // t < T+15
    wire        end_of_inference = (cycle == (t_orig + 2*DELAY_HALF - 1)); // t == T+29

    wire [N_INPUT-1:0] spike_in_gated = in_input_window
        ? spike_in
        : {N_INPUT{1'b0}};

    // ------------------------------------------------------------------
    // Delay tables (loaded from delays_layer{1,2}.hex). Each entry is a
    // raw signed-int8 byte from extract_shd_weights.py; we sign-extend
    // and compute read_offset = 15 - delay at elaboration / wire time.
    // ------------------------------------------------------------------
    reg [7:0] delays_l1_mem [0:N_INPUT-1];
    reg [7:0] delays_l2_mem [0:N_HIDDEN-1];

    initial begin
        $readmemh("delays_layer1.hex", delays_l1_mem);
        $readmemh("delays_layer2.hex", delays_l2_mem);
    end

    // ------------------------------------------------------------------
    // Layer 1 axonal delays — 140 instances, one per input neuron
    // ------------------------------------------------------------------
    wire [N_INPUT-1:0] delayed_l1;

    genvar gi;
    generate
        for (gi = 0; gi < N_INPUT; gi = gi + 1) begin: l1_delay
            wire signed [7:0] d_signed = $signed(delays_l1_mem[gi]);
            wire signed [7:0] ro_signed = 8'sd15 - d_signed;
            wire [DELAY_PTR_WIDTH-1:0] ro = ro_signed[DELAY_PTR_WIDTH-1:0];

            sc_axonal_delay #(
                .DEPTH    (MAX_DELAY),
                .PTR_WIDTH(DELAY_PTR_WIDTH)
            ) inst (
                .clk        (clk),
                .rst_n      (rst_n),
                .spike_in   (spike_in_gated[gi]),
                .read_offset(ro),
                .spike_out  (delayed_l1[gi])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Layer 1 dense (140 -> 128)
    // ------------------------------------------------------------------
    wire [N_HIDDEN*16-1:0] dense1_out_comb;

    sc_dense_int8_sparse #(
        .IN_FEATURES (N_INPUT),
        .OUT_FEATURES(N_HIDDEN),
        .WEIGHT_FILE ("weights_layer1.hex")
    ) dense1 (
        .clk                 (clk),
        .rst_n               (rst_n),
        .scale_q16_16        (scale_l1_q16_16),
        .spikes_in           (delayed_l1),
        .out_q88_packed      (/* unused */),
        .out_q88_packed_comb (dense1_out_comb)
    );

    // ------------------------------------------------------------------
    // Layer 1 Vmin_LIF — 128 instances
    // ------------------------------------------------------------------
    wire [N_HIDDEN-1:0] spikes_l1;

    generate
        for (gi = 0; gi < N_HIDDEN; gi = gi + 1) begin: l1_vmin
            wire signed [15:0] x = $signed(dense1_out_comb[16*gi +: 16]);
            sc_vmin_lif_neuron inst (
                .clk        (clk),
                .rst_n      (rst_n),
                .x_in       (x),
                .spike      (/* unused */),
                .v_out      (/* unused */),
                .spike_comb (spikes_l1[gi]),
                .v_next_comb(/* unused */)
            );
        end
    endgenerate

    // Mask layer-1 spikes after T+15 (mirrors dcls_l2 right-padding zeros)
    wire [N_HIDDEN-1:0] spikes_l1_masked = in_l1_window
        ? spikes_l1
        : {N_HIDDEN{1'b0}};

    // ------------------------------------------------------------------
    // Layer 2 axonal delays — 128 instances
    // ------------------------------------------------------------------
    wire [N_HIDDEN-1:0] delayed_l2;

    generate
        for (gi = 0; gi < N_HIDDEN; gi = gi + 1) begin: l2_delay
            wire signed [7:0] d_signed = $signed(delays_l2_mem[gi]);
            wire signed [7:0] ro_signed = 8'sd15 - d_signed;
            wire [DELAY_PTR_WIDTH-1:0] ro = ro_signed[DELAY_PTR_WIDTH-1:0];

            sc_axonal_delay #(
                .DEPTH    (MAX_DELAY),
                .PTR_WIDTH(DELAY_PTR_WIDTH)
            ) inst (
                .clk        (clk),
                .rst_n      (rst_n),
                .spike_in   (spikes_l1_masked[gi]),
                .read_offset(ro),
                .spike_out  (delayed_l2[gi])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Layer 2 dense (128 -> 128)
    // ------------------------------------------------------------------
    wire [N_HIDDEN*16-1:0] dense2_out_comb;

    sc_dense_int8_sparse #(
        .IN_FEATURES (N_HIDDEN),
        .OUT_FEATURES(N_HIDDEN),
        .WEIGHT_FILE ("weights_layer2.hex")
    ) dense2 (
        .clk                 (clk),
        .rst_n               (rst_n),
        .scale_q16_16        (scale_l2_q16_16),
        .spikes_in           (delayed_l2),
        .out_q88_packed      (/* unused */),
        .out_q88_packed_comb (dense2_out_comb)
    );

    // ------------------------------------------------------------------
    // Layer 2 Vmin_LIF — 128 instances
    // ------------------------------------------------------------------
    wire [N_HIDDEN-1:0] spikes_l2;

    generate
        for (gi = 0; gi < N_HIDDEN; gi = gi + 1) begin: l2_vmin
            wire signed [15:0] x = $signed(dense2_out_comb[16*gi +: 16]);
            sc_vmin_lif_neuron inst (
                .clk        (clk),
                .rst_n      (rst_n),
                .x_in       (x),
                .spike      (/* unused */),
                .v_out      (/* unused */),
                .spike_comb (spikes_l2[gi]),
                .v_next_comb(/* unused */)
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Layer 3 readout dense (128 -> 20) — no Vmin, voltages summed.
    // ------------------------------------------------------------------
    wire [N_OUTPUT*16-1:0] dense3_out_comb;

    sc_dense_int8_sparse #(
        .IN_FEATURES (N_HIDDEN),
        .OUT_FEATURES(N_OUTPUT),
        .WEIGHT_FILE ("weights_layer3.hex")
    ) dense3 (
        .clk                 (clk),
        .rst_n               (rst_n),
        .scale_q16_16        (scale_l3_q16_16),
        .spikes_in           (spikes_l2),
        .out_q88_packed      (/* unused */),
        .out_q88_packed_comb (dense3_out_comb)
    );

    // ------------------------------------------------------------------
    // 20-class voltage accumulator — sums dense3_out_comb every cycle
    // while `running == 1`. The result is exposed as a packed signed
    // 32-bit bus so the testbench can read it word-by-word for cosim.
    // ------------------------------------------------------------------
    integer ai;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle               <= 16'd0;
            running             <= 1'b0;
            done                <= 1'b0;
            output_v_sum_packed <= {(20*32){1'b0}};
        end else begin
            done <= 1'b0;
            if (start) begin
                // Start-cycle: this is iter 0 — combinational chain has
                // already produced dense3_out_comb for it. Seed the
                // accumulator with that contribution and advance `cycle`
                // so the NEXT posedge sees `cycle = 1` (= iter 1 about
                // to be processed). Python's run_inference_q88 includes
                // the first iter in its sum the same way.
                cycle   <= 16'd1;
                running <= 1'b1;
                for (ai = 0; ai < N_OUTPUT; ai = ai + 1) begin
                    output_v_sum_packed[32*ai +: 32] <=
                        $signed({{16{dense3_out_comb[16*ai+15]}},
                                  dense3_out_comb[16*ai +: 16]});
                end
            end else if (running) begin
                for (ai = 0; ai < N_OUTPUT; ai = ai + 1) begin
                    output_v_sum_packed[32*ai +: 32] <=
                        $signed(output_v_sum_packed[32*ai +: 32])
                        + $signed({{16{dense3_out_comb[16*ai+15]}},
                                   dense3_out_comb[16*ai +: 16]});
                end
                cycle <= cycle + 16'd1;
                if (end_of_inference) begin
                    running <= 1'b0;
                    done    <= 1'b1;
                end
            end
        end
    end

endmodule
