// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Top-level SHD inference network (Masquelier model, pipelined)
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
// run the network for `T_orig + 30` logical iterations per inference and
// mask the layer-1 spike output to zero from iter `T_orig + 15` onwards
// (the equivalent of the dcls_l2 right-padding zeros that PyTorch sees).
//
// Pipelining
// ----------
//
// The chain is broken by three pipeline registers — one after each dense
// layer — giving a 3-cycle latency from input to accumulator. The
// registers are the existing `sc_dense_int8_sparse` registered output
// ports (`out_q88_packed`); the combinational taps (`out_q88_packed_comb`)
// used by the older single-cycle variant are no longer wired.
//
//   clock C   | Stage 0 iter | Stage 1 iter | Stage 2 iter | accumulator
//   ----------+--------------+--------------+--------------+-------------
//   0 (start) | 0            | (reset)      | (reset)      | (reset)
//   1         | 1            | 0            | (reset)      | (reset)
//   2         | 2            | 1            | 0            | (reset)
//   3         | 3            | 2            | 1            | iter 0
//   ...       | ...          | ...          | ...          | ...
//   T2+2      | T2+2         | T2+1         | T2           | iter T2-1
//
// Where T2 = T_orig + 30. Total clocks per inference = T2 + 3.
//
// Stateful cells in stages 1/2 (vmin_lif_l1, axon_delay_l2, vmin_lif_l2)
// must NOT update their state during the pipeline-fill clocks (0/1/2)
// because their combinational inputs carry reset-zero filler data, not a
// valid iteration. We hold them in async reset via a gated rst_n signal:
//
//   rst_n_stage1 = rst_n & pipe1_active   // high from clock 1 onwards
//   rst_n_stage2 = rst_n & pipe2_active   // high from clock 2 onwards
//
// The `pipe{1,2,3}_active` registers are zero at start, then shift in a 1
// each subsequent clock so stage N is valid from clock N. The accumulator
// uses `pipe3_active` as an explicit enable rather than a reset gate.
//
// The pipeline-register approach is bit-true against the combinational
// predecessor (and against `tools/shd_q88_reference.py::run_inference_q88`)
// because each iter's compute happens with exactly the same operands as
// the unpipelined version; only the TIMING shifts by the pipeline depth.
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
    localparam integer PIPELINE_DEPTH = 3;                // dense1 + dense2 + dense3 regs

    // ------------------------------------------------------------------
    // Cycle counter and pipeline stage valids
    // ------------------------------------------------------------------
    // `cycle` counts the number of clocks since the start strobe, inclusive.
    // Clock 0 is the start posedge (stage 0 processes iter 0), clock T2+2
    // is the last accumulator update (iter T2-1 reaches the accumulator).
    // `pipe{1,2,3}_active` enable stateful cells in later stages only after
    // enough clocks have passed to fill the pipeline up to that stage.
    reg  [15:0] cycle;
    reg         pipe1_active;   // stage 1 (vmin_lif_l1, axon_delay_l2) valid from clock 1
    reg         pipe2_active;   // stage 2 (vmin_lif_l2) valid from clock 2
    reg         pipe3_active;   // stage 3 (accumulator) valid from clock 3

    wire rst_n_stage1 = rst_n & pipe1_active;
    wire rst_n_stage2 = rst_n & pipe2_active;

    // Gating signals. Each stage reads its own iter index; since the
    // pipeline shifts iter C at stage 0 to iter (C-N) at stage N, and
    // `cycle` IS the clock index of stage 0 at this posedge, we can
    // compute the gate for each stage off `cycle` directly.
    //
    //   stage 0 gate: `in_input_window = cycle < t_orig`
    //                 (input feed runs only while iter index < T_orig)
    //
    //   stage 1 gate: `in_l1_window = cycle < (t_orig + DELAY_HALF + 1)`
    //                 stage 1 processes iter (cycle-1); the Python reference
    //                 masks stage-1 output for iters >= T_orig + DELAY_HALF
    //                 so the Verilog gate is (cycle-1) < T_orig+15 ⇒
    //                 cycle < T_orig + 16.
    wire in_input_window = (cycle < t_orig);
    wire in_l1_window    = (cycle < (t_orig + DELAY_HALF + 1));

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

    // Delay initialisation: only performed during simulation. Synthesis
    // tools (yosys, Vivado) define `SYNTHESIS` so the `$readmemh` calls
    // are skipped — the delay tables become uninitialised memories that
    // the real bitstream writes through AXI at boot time.
    initial begin
`ifndef SYNTHESIS
        $readmemh("delays_layer1.hex", delays_l1_mem);
        $readmemh("delays_layer2.hex", delays_l2_mem);
`endif
    end

    // ------------------------------------------------------------------
    // Layer 1 axonal delays — 140 instances, one per input neuron
    // (stage 0 — always active, not gated by pipeline valid)
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
    // Layer 1 dense (140 -> 128) — REGISTERED output (pipeline stage 0 -> 1)
    // ------------------------------------------------------------------
    wire [N_HIDDEN*16-1:0] dense1_out_reg;

    sc_dense_int8_sparse #(
        .IN_FEATURES (N_INPUT),
        .OUT_FEATURES(N_HIDDEN),
        .WEIGHT_FILE ("weights_layer1.hex")
    ) dense1 (
        .clk                 (clk),
        .rst_n               (rst_n),
        .scale_q16_16        (scale_l1_q16_16),
        .spikes_in           (delayed_l1),
        .out_q88_packed      (dense1_out_reg),    // registered → pipeline reg
        .out_q88_packed_comb (/* unused */)
    );

    // ------------------------------------------------------------------
    // Layer 1 Vmin_LIF — 128 instances (stage 1, gated by pipe1_active)
    // ------------------------------------------------------------------
    wire [N_HIDDEN-1:0] spikes_l1;

    generate
        for (gi = 0; gi < N_HIDDEN; gi = gi + 1) begin: l1_vmin
            wire signed [15:0] x = $signed(dense1_out_reg[16*gi +: 16]);
            sc_vmin_lif_neuron inst (
                .clk        (clk),
                .rst_n      (rst_n_stage1),
                .x_in       (x),
                .spike      (/* unused */),
                .v_out      (/* unused */),
                .spike_comb (spikes_l1[gi]),
                .v_next_comb(/* unused */)
            );
        end
    endgenerate

    // Mask layer-1 spikes after T+15 (mirrors dcls_l2 right-padding zeros).
    // The gate uses the pipelined `in_l1_window` definition — cycle < T+16
    // — because stage 1 processes iter (cycle-1) at this posedge.
    wire [N_HIDDEN-1:0] spikes_l1_masked = in_l1_window
        ? spikes_l1
        : {N_HIDDEN{1'b0}};

    // ------------------------------------------------------------------
    // Layer 2 axonal delays — 128 instances (stage 1, gated by pipe1_active)
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
                .rst_n      (rst_n_stage1),
                .spike_in   (spikes_l1_masked[gi]),
                .read_offset(ro),
                .spike_out  (delayed_l2[gi])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Layer 2 dense (128 -> 128) — REGISTERED output (pipeline stage 1 -> 2)
    // ------------------------------------------------------------------
    wire [N_HIDDEN*16-1:0] dense2_out_reg;

    sc_dense_int8_sparse #(
        .IN_FEATURES (N_HIDDEN),
        .OUT_FEATURES(N_HIDDEN),
        .WEIGHT_FILE ("weights_layer2.hex")
    ) dense2 (
        .clk                 (clk),
        .rst_n               (rst_n),
        .scale_q16_16        (scale_l2_q16_16),
        .spikes_in           (delayed_l2),
        .out_q88_packed      (dense2_out_reg),
        .out_q88_packed_comb (/* unused */)
    );

    // ------------------------------------------------------------------
    // Layer 2 Vmin_LIF — 128 instances (stage 2, gated by pipe2_active)
    // ------------------------------------------------------------------
    wire [N_HIDDEN-1:0] spikes_l2;

    generate
        for (gi = 0; gi < N_HIDDEN; gi = gi + 1) begin: l2_vmin
            wire signed [15:0] x = $signed(dense2_out_reg[16*gi +: 16]);
            sc_vmin_lif_neuron inst (
                .clk        (clk),
                .rst_n      (rst_n_stage2),
                .x_in       (x),
                .spike      (/* unused */),
                .v_out      (/* unused */),
                .spike_comb (spikes_l2[gi]),
                .v_next_comb(/* unused */)
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Layer 3 readout dense (128 -> 20) — REGISTERED output (pipeline stage 2 -> 3)
    // ------------------------------------------------------------------
    wire [N_OUTPUT*16-1:0] dense3_out_reg;

    sc_dense_int8_sparse #(
        .IN_FEATURES (N_HIDDEN),
        .OUT_FEATURES(N_OUTPUT),
        .WEIGHT_FILE ("weights_layer3.hex")
    ) dense3 (
        .clk                 (clk),
        .rst_n               (rst_n),
        .scale_q16_16        (scale_l3_q16_16),
        .spikes_in           (spikes_l2),
        .out_q88_packed      (dense3_out_reg),
        .out_q88_packed_comb (/* unused */)
    );

    // ------------------------------------------------------------------
    // 20-class voltage accumulator and pipeline valid bookkeeping.
    //
    // At the start posedge, cycle jumps to 1 and running<=1; pipeline
    // valids stay 0. Each subsequent clock the valids shift in a 1 so
    // stage N becomes active one clock later than stage N-1. The
    // accumulator updates whenever `pipe3_active` is set (from clock 3),
    // and the `end_of_inference` condition stops the run after the last
    // iter (T2-1) has reached the accumulator at cycle = T2 + 2.
    // ------------------------------------------------------------------
    integer ai;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running             <= 1'b0;
            done                <= 1'b0;
            cycle               <= 16'd0;
            pipe1_active        <= 1'b0;
            pipe2_active        <= 1'b0;
            pipe3_active        <= 1'b0;
            output_v_sum_packed <= {(N_OUTPUT*32){1'b0}};
        end else begin
            done <= 1'b0;
            if (!running && start) begin
                // At end of the start posedge we want stage 1 to come out
                // of reset immediately so clock 1 can process iter 0 (the
                // value dense1 just latched into `dense1_out_reg`). Setting
                // `pipe1_active <= 1` here yields POST=1 so rst_n_stage1
                // is high by PRE of clock 1. Stages 2/3 keep their 1-clock
                // shift behind stage 1 via the running branch below.
                running             <= 1'b1;
                cycle               <= 16'd1;
                pipe1_active        <= 1'b1;    // active from PRE of clock 1
                pipe2_active        <= 1'b0;
                pipe3_active        <= 1'b0;
                output_v_sum_packed <= {(N_OUTPUT*32){1'b0}};
            end else if (running) begin
                pipe1_active <= 1'b1;            // latched high, stays
                pipe2_active <= pipe1_active;    // high from PRE of clock 2
                pipe3_active <= pipe2_active;    // high from PRE of clock 3

                if (pipe3_active) begin
                    for (ai = 0; ai < N_OUTPUT; ai = ai + 1) begin
                        output_v_sum_packed[32*ai +: 32] <=
                            $signed(output_v_sum_packed[32*ai +: 32])
                            + $signed({{16{dense3_out_reg[16*ai+15]}},
                                       dense3_out_reg[16*ai +: 16]});
                    end
                end

                if (cycle == (t_orig + 2*DELAY_HALF + 2)) begin
                    running <= 1'b0;
                    done    <= 1'b1;
                end else begin
                    cycle <= cycle + 16'd1;
                end
            end
        end
    end

endmodule
