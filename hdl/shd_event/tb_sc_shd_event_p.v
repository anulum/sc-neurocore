// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Zero-branch RTL testbench for sc_shd_event_p
//
// Derived from the event-arm path of the group's earlier design study testbench.
// Loads q1.hex, q2.hex, q3.hex, d1.hex, d2.hex through the idle group-word load port,
// streams samples from stimulus.hex, and writes outputs.txt with the 20 class accumulators
// followed by '|' and the 16 event-arm counters.
`timescale 1ns / 1ps

module tb_sc_shd_event_p;
    parameter integer FRAC_BITS = 0;
    parameter integer THETA1 = 1;
    parameter integer THETA2 = 1;
    parameter integer V1_WIDTH = 22;
    parameter integer V2_WIDTH = 17;
    parameter integer WSHAPE = 6;
    parameter integer CAP1 = 140;
    parameter integer CAP2 = 128;
    parameter integer QSHAPE = 12;
    parameter integer P = 8;
    parameter integer MAX_WORDS = 4000000;

    reg clk = 0, rst_n = 0, start = 0, frame_valid = 0;
    reg load_we = 0;
    reg [2:0] load_sel = 0;
    reg [7:0] load_addr = 0;
    reg [6:0] load_grp = 0;
    reg [1023:0] load_data = 0;
    reg [15:0] t_orig = 0;
    reg [559:0] frame = 0;
    wire frame_ready, busy, done, error, overflow;
    wire [639:0] out_packed;

    sc_shd_event_p #(
        .P(P),
        .FRAC_BITS(FRAC_BITS),
        .THETA1(THETA1),
        .THETA2(THETA2),
        .V1_WIDTH(V1_WIDTH),
        .V2_WIDTH(V2_WIDTH),
        .CAP1(CAP1),
        .CAP2(CAP2),
        .WSHAPE(WSHAPE),
        .QSHAPE(QSHAPE)
    ) dut (
        .overflow(overflow),
        .load_grp(load_grp),
        .clk(clk),
        .rst_n(rst_n),
        .load_we(load_we),
        .load_sel(load_sel),
        .load_addr(load_addr),
        .load_data(load_data[8*P-1:0]),
        .start(start),
        .t_orig(t_orig),
        .frame(frame),
        .frame_valid(frame_valid),
        .frame_ready(frame_ready),
        .busy(busy),
        .done(done),
        .error(error),
        .out_packed(out_packed)
    );

    parameter real HALF_PERIOD = 5.0;
    always #(HALF_PERIOD) clk = ~clk;

    reg [559:0]  stim [0:MAX_WORDS-1];
    reg [1023:0] q1 [0:139];
    reg [1023:0] q2 [0:127];
    reg [159:0]  q3 [0:127];
    reg [4:0]    d1 [0:139];
    reg [4:0]    d2 [0:127];
    integer fd, n, s, f, k, ptr, r, gq;

    task load_word(input [2:0] sel, input [7:0] addr, input [6:0] grp, input [1023:0] data);
        begin
            @(negedge clk);
            load_we = 1;
            load_sel = sel;
            load_addr = addr;
            load_grp = grp;
            load_data = data;
            @(negedge clk);
            load_we = 0;
        end
    endtask

    initial begin
        $readmemh("stimulus.hex", stim);
        $readmemh("q1.hex", q1);
        $readmemh("q2.hex", q2);
        $readmemh("q3.hex", q3);
        $readmemh("d1.hex", d1);
        $readmemh("d2.hex", d2);
        fd = $fopen("outputs.txt", "w");
        repeat (3) @(posedge clk);
        rst_n = 1;
        for (r = 0; r < 140; r = r + 1)
            for (gq = 0; gq < 128 / P; gq = gq + 1)
                load_word(0, r[7:0], gq[6:0], q1[r] >> (gq * 8 * P));
        for (r = 0; r < 128; r = r + 1)
            for (gq = 0; gq < 128 / P; gq = gq + 1)
                load_word(1, r[7:0], gq[6:0], q2[r] >> (gq * 8 * P));
        for (r = 0; r < 128; r = r + 1)
            for (gq = 0; gq < (20 + P - 1) / P; gq = gq + 1)
                load_word(2, r[7:0], gq[6:0], {864'd0, q3[r]} >> (gq * 8 * P));
        for (r = 0; r < 140; r = r + 1)
            load_word(3, r[7:0], 7'd0, {1019'd0, d1[r]});
        for (r = 0; r < 128; r = r + 1)
            load_word(4, r[7:0], 7'd0, {1019'd0, d2[r]});
        @(posedge clk);
        n = stim[0][31:0];
        ptr = 1;
        for (s = 0; s < n; s = s + 1) begin
            t_orig = stim[ptr][15:0];
            ptr = ptr + 1;
            dut.w1.reads = 0;
            dut.w1.writes = 0;
            dut.w2.reads = 0;
            dut.w2.writes = 0;
            dut.q1.reads = 0;
            dut.q2.reads = 0;
            dut.q3.reads = 0;
            @(negedge clk);
            start = 1;
            @(negedge clk);
            start = 0;
            if (t_orig == 0 || t_orig > 1024) begin
                @(negedge clk);
                $fwrite(fd, "REFUSED %0d\n", error);
                $fflush(fd);
            end else begin
                f = 0;
                while (!done) begin
                    @(negedge clk);
                    if (frame_ready && f < t_orig) begin
                        frame = stim[ptr];
                        frame_valid = 1;
                    end else begin
                        frame_valid = 0;
                    end
                    @(posedge clk);
                    #1;
                    if (frame_valid && frame_ready === 1'b0) begin
                        f = f + 1;
                        ptr = ptr + 1;
                        frame_valid = 0;
                    end
                end
                for (k = 0; k < 20; k = k + 1)
                    $fwrite(fd, "%0d ", $signed(out_packed[32*k +: 32]));
                $fwrite(fd, "| %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                        dut.cycles, dut.l1_rows, dut.l2_rows, dut.l3_rows,
                        dut.mem_rw, dut.leak_it,
                        dut.w1.reads, dut.w1.writes, dut.w2.reads, dut.w2.writes,
                        dut.q1.reads, dut.q2.reads, dut.q3.reads,
                        dut.pushes, dut.pops, overflow);
                $fflush(fd);
            end
        end
        $fclose(fd);
        $finish;
    end
endmodule
