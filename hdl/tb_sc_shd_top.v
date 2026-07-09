// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for sc_shd_top.v
//
// File-based cosim driver for the end-to-end SHD inference network.
//
// The Python harness writes the following files into a working directory
// then runs vvp from there so that all `$readmemh` paths resolve:
//
//   weights_layer1.hex   one signed-int8 byte per line, 128 * 140 entries
//   weights_layer2.hex   one signed-int8 byte per line, 128 * 128 entries
//   weights_layer3.hex   one signed-int8 byte per line,  20 * 128 entries
//   delays_layer1.hex    one signed-int8 byte per line, 140 entries
//   delays_layer2.hex    one signed-int8 byte per line, 128 entries
//   spikes.hex           one 140-bit hex per line, T_orig samples
//   scales.txt           three signed Q16.16 ints, one per line
//
// Plus-args:
//   +T=<num_samples>          number of input timesteps to drive
//
// Output:
//   outputs.txt              one signed-int32 per line, 20 entries
//                            (the final output_v_sum_packed contents)

`timescale 1ns / 1ps

module tb_sc_shd_top;

    localparam integer N_INPUT  = 140;
    localparam integer N_OUTPUT = 20;
    localparam integer MAX_INPUT_LEN = 1024;

    reg                 clk;
    reg                 rst_n;
    reg                 start;
    reg  [15:0]         t_orig;
    reg  [N_INPUT-1:0]  spike_in;
    reg  signed [31:0]  scale_l1_q16_16;
    reg  signed [31:0]  scale_l2_q16_16;
    reg  signed [31:0]  scale_l3_q16_16;
    wire                running;
    wire                done;
    wire signed [N_OUTPUT*32-1:0] output_v_sum_packed;

    sc_shd_top dut (
        .clk                 (clk),
        .rst_n               (rst_n),
        .start               (start),
        .t_orig              (t_orig),
        .spike_in            (spike_in),
        .scale_l1_q16_16     (scale_l1_q16_16),
        .scale_l2_q16_16     (scale_l2_q16_16),
        .scale_l3_q16_16     (scale_l3_q16_16),
        .running             (running),
        .done                (done),
        .output_v_sum_packed (output_v_sum_packed)
    );

    // Stimulus storage
    reg [N_INPUT-1:0] stim_mem [0:MAX_INPUT_LEN-1];

    integer t_in;
    integer scale_fd;
    integer s1_int, s2_int, s3_int;
    integer out_fd;
    integer i;
    integer j;
    reg signed [31:0] sum_word;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        t_in = 0;
        if (!$value$plusargs("T=%d", t_in)) t_in = 0;
        if (t_in <= 0 || t_in > MAX_INPUT_LEN) begin
            $display("ERROR: invalid +T=%0d (must be 1..%0d)",
                     t_in, MAX_INPUT_LEN);
            $finish;
        end

        scale_fd = $fopen("scales.txt", "r");
        if (scale_fd == 0) begin
            $display("ERROR: cannot open scales.txt");
            $finish;
        end
        if ($fscanf(scale_fd, "%d %d %d", s1_int, s2_int, s3_int) != 3) begin
            $display("ERROR: cannot parse scales.txt");
            $finish;
        end
        $fclose(scale_fd);
        scale_l1_q16_16 = s1_int;
        scale_l2_q16_16 = s2_int;
        scale_l3_q16_16 = s3_int;

        $readmemh("spikes.hex", stim_mem, 0, t_in - 1);

        // Initialise control signals
        rst_n    = 1'b0;
        start    = 1'b0;
        t_orig   = t_in[15:0];
        spike_in = {N_INPUT{1'b0}};

        // Reset hold
        repeat (3) @(posedge clk);

        // Release reset between posedges
        @(negedge clk);
        rst_n = 1'b1;

        // Issue start strobe with sample[0] already on the bus, mirroring
        // the race-free idiom used by the unit-level testbenches.
        @(negedge clk);
        spike_in = stim_mem[0];
        start    = 1'b1;
        @(posedge clk);

        // Drop start, drive remaining samples while `running == 1`. The
        // upper bound `t_in + 33` covers:
        //   - t_in - 1 remaining real samples (iters 1 .. t_in-1)
        //   - 30 zero-pad iters for the two DCLS asymmetric-padding tails
        //   - 3 extra drain clocks for the 3-stage pipeline (see sc_shd_top.v)
        // The DUT asserts `done` at cycle == t_orig + 2*DELAY_HALF + 2 which
        // falls inside this loop for every valid T.
        @(negedge clk);
        start = 1'b0;
        for (i = 1; i < t_in + 33; i = i + 1) begin
            if (i < t_in)
                spike_in = stim_mem[i];
            else
                spike_in = {N_INPUT{1'b0}};
            @(posedge clk);
            @(negedge clk);
        end

        // Wait for done strobe
        wait (done);
        @(posedge clk);

        out_fd = $fopen("outputs.txt", "w");
        if (out_fd == 0) begin
            $display("ERROR: cannot open outputs.txt");
            $finish;
        end
        $fwrite(out_fd, "# class output_v_sum\n");
        for (j = 0; j < N_OUTPUT; j = j + 1) begin
            sum_word = $signed(output_v_sum_packed[32*j +: 32]);
            $fwrite(out_fd, "%0d %0d\n", j, sum_word);
        end
        $fclose(out_fd);

        $display("tb_sc_shd_top: T=%0d, output ready", t_in);
        $finish;
    end

    initial begin
        if ($test$plusargs("VCD")) begin
            $dumpfile("tb_sc_shd_top.vcd");
            $dumpvars(0, tb_sc_shd_top);
        end
    end

endmodule
