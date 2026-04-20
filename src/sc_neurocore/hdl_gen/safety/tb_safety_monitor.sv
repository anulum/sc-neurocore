// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for Neuro-Safe Hardware Monitor
//
// Verifies all 6 formally proven safety properties.

`timescale 1ns / 1ps

module tb_safety_monitor;

    reg         clk;
    reg         rst_n;
    reg  [15:0] probe_current;
    reg  [15:0] probe_voltage;
    reg  [15:0] probe_coherence;
    reg  [15:0] probe_popcount_k;
    reg  [15:0] probe_sc_add_result;
    reg  [15:0] probe_membrane;
    reg  signed [15:0] probe_scc_numer;
    reg  [15:0] probe_scc_denom;
    wire        hardware_halt;
    wire [5:0]  violation_flags;

    neuro_safe_monitor #(
        .MAX_CURRENT(16'h7FFF),
        .MAX_VOLTAGE(16'hC000),
        .COHERENCE_LIMIT(16'h0100),
        .SC_DENOM(16'h0100),
        .LIF_V_MAX(16'hC000)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .probe_current(probe_current),
        .probe_voltage(probe_voltage),
        .probe_coherence(probe_coherence),
        .probe_popcount_k(probe_popcount_k),
        .probe_sc_add_result(probe_sc_add_result),
        .probe_membrane(probe_membrane),
        .probe_scc_numer(probe_scc_numer),
        .probe_scc_denom(probe_scc_denom),
        .hardware_halt(hardware_halt),
        .violation_flags(violation_flags)
    );

    // Clock generation: 100 MHz
    always #5 clk = ~clk;

    integer pass_count;
    integer fail_count;

    task assert_no_halt(input [127:0] label);
        begin
            @(posedge clk); #1;
            if (hardware_halt !== 0) begin
                $display("FAIL [%0s]: halt asserted unexpectedly, flags=%b", label, violation_flags);
                fail_count = fail_count + 1;
            end else begin
                $display("PASS [%0s]: no halt", label);
                pass_count = pass_count + 1;
            end
        end
    endtask

    task assert_halt(input [127:0] label, input [5:0] expected_flag);
        begin
            @(posedge clk); #1;
            if (hardware_halt !== 1) begin
                $display("FAIL [%0s]: halt NOT asserted, flags=%b", label, violation_flags);
                fail_count = fail_count + 1;
            end else if ((violation_flags & expected_flag) == 0) begin
                $display("FAIL [%0s]: wrong flag, expected %b got %b", label, expected_flag, violation_flags);
                fail_count = fail_count + 1;
            end else begin
                $display("PASS [%0s]: halt asserted, flag=%b", label, violation_flags);
                pass_count = pass_count + 1;
            end
        end
    endtask

    task reset_dut;
        begin
            rst_n = 0;
            probe_current = 0;
            probe_voltage = 0;
            probe_coherence = 16'hFFFF;
            probe_popcount_k = 0;
            probe_sc_add_result = 0;
            probe_membrane = 0;
            probe_scc_numer = 0;
            probe_scc_denom = 16'h0100;
            @(posedge clk);
            @(posedge clk);
            rst_n = 1;
            @(posedge clk);
        end
    endtask

    initial begin
        $dumpfile("tb_safety_monitor.vcd");
        $dumpvars(0, tb_safety_monitor);

        clk = 0;
        pass_count = 0;
        fail_count = 0;

        // ── Test 1: Normal operation — no violations ──
        reset_dut;
        probe_current   = 16'h1000;
        probe_voltage   = 16'h2000;
        probe_coherence = 16'hFFFF;
        probe_popcount_k = 16'h0080;
        probe_sc_add_result = 16'h0080;
        probe_membrane  = 16'h4000;
        probe_scc_numer = 16'h0050;
        probe_scc_denom = 16'h0100;
        assert_no_halt("T1: normal operation");

        // ── Test 2: [P1] Current overflow ──
        reset_dut;
        probe_current = 16'hFFFF;
        assert_halt("T2: P1 current overflow", 6'b000001);

        // ── Test 3: [P1] Voltage overflow ──
        reset_dut;
        probe_voltage = 16'hFFFF;
        assert_halt("T3: P1 voltage overflow", 6'b000001);

        // ── Test 4: [P1] Coherence below limit ──
        reset_dut;
        probe_coherence = 16'h0010;
        assert_halt("T4: P1 coherence violation", 6'b000001);

        // ── Test 5: [P2] Monotone coherence violation ──
        reset_dut;
        probe_coherence = 16'hF000;
        @(posedge clk); #1;
        probe_coherence = 16'h0F00;  // decreased!
        assert_halt("T5: P2 monotone violation", 6'b000010);

        // ── Test 6: [P3] Popcount exceeds bitstream length ──
        reset_dut;
        probe_popcount_k = 16'h0200;  // > SC_DENOM(0x0100)
        assert_halt("T6: P3 precision violation", 6'b000100);

        // ── Test 7: [P4] SC addition exceeds denominator ──
        reset_dut;
        probe_sc_add_result = 16'h0200;  // > SC_DENOM
        assert_halt("T7: P4 range violation", 6'b001000);

        // ── Test 8: [P5] Membrane exceeds V_max ──
        reset_dut;
        probe_membrane = 16'hFFFF;  // > LIF_V_MAX
        assert_halt("T8: P5 membrane violation", 6'b010000);

        // ── Test 9: [P6] SCC numerator out of range (positive) ──
        reset_dut;
        probe_scc_numer = 16'h0200;  // > denom(0x0100)
        probe_scc_denom = 16'h0100;
        assert_halt("T9: P6 SCC positive overflow", 6'b100000);

        // ── Test 10: [P6] SCC numerator out of range (negative) ──
        reset_dut;
        probe_scc_numer = -16'sd512;  // |-512| > 256
        probe_scc_denom = 16'h0100;
        assert_halt("T10: P6 SCC negative overflow", 6'b100000);

        // ── Test 11: Boundary — exact limits (no violation) ──
        // Set coherence to target value DURING reset so monotone check is clean
        rst_n = 0;
        probe_current        = 16'h7FFF;  // == MAX_CURRENT
        probe_voltage        = 16'hC000;  // == MAX_VOLTAGE
        probe_coherence      = 16'h0100;  // == COHERENCE_LIMIT (set before rst_n rises)
        probe_popcount_k     = 16'h0100;  // == SC_DENOM
        probe_sc_add_result  = 16'h0100;  // == SC_DENOM
        probe_membrane       = 16'hC000;  // == LIF_V_MAX
        probe_scc_numer      = 16'h0100;  // == denom
        probe_scc_denom      = 16'h0100;
        @(posedge clk); @(posedge clk);
        rst_n = 1;
        @(posedge clk); // settle prev_coherence
        assert_no_halt("T11: exact boundaries");

        // ── Summary ──
        $display("");
        $display("========================================");
        $display("  Results: %0d passed, %0d failed", pass_count, fail_count);
        $display("========================================");

        if (fail_count > 0) begin
            $display("FAIL: %0d tests failed", fail_count);
        end else begin
            $display("ALL PASS");
        end

        $finish;
    end

endmodule
