# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - AER priority queue behavioural tests.

"""Behavioural contract for the AER priority queue workflow.

This is the hardware-control workflow contract for NEU-C.4: urgent AER events
must overtake best-effort events under downstream backpressure while preserving
FIFO order inside each priority class and latching safety violations.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sc_neurocore.hdl import AERPriorityEvent, AERPriorityQueueReference


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_priority_queue_serves_critical_before_best_effort() -> None:
    queue = AERPriorityQueueReference(capacity=8)
    events = [
        AERPriorityEvent(target_id=1, weight=10, timestamp=0, priority=3),
        AERPriorityEvent(target_id=2, weight=20, timestamp=1, priority=0),
        AERPriorityEvent(target_id=3, weight=30, timestamp=2, priority=1),
        AERPriorityEvent(target_id=4, weight=40, timestamp=3, priority=0),
    ]

    assert queue.extend(events) == len(events)

    drained = queue.drain()
    assert [event.target_id for event in drained] == [2, 4, 3, 1]
    assert [event.priority for event in drained] == [0, 0, 1, 3]


def test_priority_queue_preserves_fifo_inside_priority_class() -> None:
    queue = AERPriorityQueueReference(capacity=8)
    events = [
        AERPriorityEvent(target_id=1, weight=10, timestamp=0, priority=2),
        AERPriorityEvent(target_id=2, weight=20, timestamp=1, priority=2),
        AERPriorityEvent(target_id=3, weight=30, timestamp=2, priority=2),
    ]

    queue.extend(events)

    assert queue.drain() == events


def test_priority_queue_backpressure_latches_drop() -> None:
    queue = AERPriorityQueueReference(capacity=2)

    assert queue.enqueue(AERPriorityEvent(target_id=1, weight=1, timestamp=0, priority=1))
    assert queue.enqueue(AERPriorityEvent(target_id=2, weight=2, timestamp=1, priority=1))
    assert not queue.enqueue(AERPriorityEvent(target_id=3, weight=3, timestamp=2, priority=1))

    assert queue.full
    assert queue.dropped_event


def test_priority_queue_deadline_trap_requires_output_stall() -> None:
    queue = AERPriorityQueueReference(capacity=4, max_latency_cycles=1)
    queue.enqueue(AERPriorityEvent(target_id=7, weight=4, timestamp=0, priority=0))

    first = queue.step(None, out_ready=False)
    second = queue.step(None, out_ready=False)

    assert first.critical_deadline_violation is False
    assert second.critical_deadline_violation is True
    assert queue.step(None, out_ready=True).output == AERPriorityEvent(
        target_id=7,
        weight=4,
        timestamp=0,
        priority=0,
    )


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=15),
            st.integers(min_value=-64, max_value=63),
            st.integers(min_value=0, max_value=255),
            st.integers(min_value=0, max_value=3),
        ),
        min_size=1,
        max_size=24,
    )
)
def test_priority_queue_randomized_order_matches_priority_fifo(
    raw_events: list[tuple[int, int, int, int]],
) -> None:
    events = [
        AERPriorityEvent(target_id=target, weight=weight, timestamp=timestamp, priority=priority)
        for target, weight, timestamp, priority in raw_events
    ]
    queue = AERPriorityQueueReference(capacity=len(events))

    assert queue.extend(events) == len(events)

    expected = [
        event
        for _, event in sorted(enumerate(events), key=lambda item: (item[1].priority, item[0]))
    ]
    assert queue.drain() == expected


def test_hdl_priority_queue_simulates_backpressure_contract(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("Icarus Verilog is required for the HDL queue simulation")

    testbench = tmp_path / "tb_sc_aer_priority_queue.v"
    output = tmp_path / "tb_sc_aer_priority_queue.out"
    testbench.write_text(
        textwrap.dedent(
            r"""
            `timescale 1ns/1ps
            `default_nettype none

            module tb_sc_aer_priority_queue;
                reg clk = 1'b0;
                reg rst_n = 1'b0;
                reg in_valid = 1'b0;
                wire in_ready;
                reg [3:0] in_target_id = 4'd0;
                reg signed [7:0] in_weight = 8'sd0;
                reg [7:0] in_timestamp = 8'd0;
                reg [1:0] in_priority = 2'd0;
                wire out_valid;
                reg out_ready = 1'b0;
                wire [3:0] out_target_id;
                wire signed [7:0] out_weight;
                wire [7:0] out_timestamp;
                wire [1:0] out_priority;
                wire full;
                wire empty;
                wire busy;
                wire dropped_event;
                wire critical_deadline_violation;
                wire [2:0] occupancy;

                always #5 clk = ~clk;

                sc_aer_priority_queue #(
                    .NEURON_ID_WIDTH(4),
                    .DATA_WIDTH(8),
                    .TIMESTAMP_WIDTH(8),
                    .PRIO_WIDTH(2),
                    .QUEUE_DEPTH(4),
                    .MAX_LATENCY_CYCLES(1)
                ) dut (
                    .clk(clk),
                    .rst_n(rst_n),
                    .in_valid(in_valid),
                    .in_ready(in_ready),
                    .in_target_id(in_target_id),
                    .in_weight(in_weight),
                    .in_timestamp(in_timestamp),
                    .in_priority(in_priority),
                    .out_valid(out_valid),
                    .out_ready(out_ready),
                    .out_target_id(out_target_id),
                    .out_weight(out_weight),
                    .out_timestamp(out_timestamp),
                    .out_priority(out_priority),
                    .full(full),
                    .empty(empty),
                    .busy(busy),
                    .dropped_event(dropped_event),
                    .critical_deadline_violation(critical_deadline_violation),
                    .occupancy(occupancy)
                );

                task push_event;
                    input [3:0] target;
                    input [1:0] prio;
                    begin
                        @(negedge clk);
                        in_target_id = target;
                        in_weight = {4'd0, target};
                        in_timestamp = {4'd0, target};
                        in_priority = prio;
                        in_valid = 1'b1;
                        @(negedge clk);
                        in_valid = 1'b0;
                    end
                endtask

                initial begin
                    #12;
                    rst_n = 1'b1;
                    out_ready = 1'b0;

                    push_event(4'd1, 2'd3);
                    push_event(4'd2, 2'd0);
                    @(posedge clk); #1;
                    if (!out_valid || out_target_id !== 4'd2 || out_priority !== 2'd0) begin
                        $fatal(1, "critical event did not overtake best-effort event");
                    end
                    @(posedge clk); #1;
                    if (!critical_deadline_violation) begin
                        $fatal(1, "critical deadline trap did not latch under backpressure");
                    end

                    @(negedge clk);
                    out_ready = 1'b1;
                    @(posedge clk); #1;
                    if (!out_valid || out_target_id !== 4'd1 || out_priority !== 2'd3) begin
                        $fatal(1, "best-effort event was not preserved after critical drain");
                    end
                    @(posedge clk); #1;
                    if (!empty) begin
                        $fatal(1, "queue did not drain after both events were accepted");
                    end

                    @(negedge clk);
                    out_ready = 1'b0;
                    push_event(4'd3, 2'd1);
                    push_event(4'd4, 2'd1);
                    push_event(4'd5, 2'd1);
                    push_event(4'd6, 2'd1);
                    @(negedge clk);
                    in_target_id = 4'd7;
                    in_weight = 8'sd7;
                    in_timestamp = 8'd7;
                    in_priority = 2'd1;
                    in_valid = 1'b1;
                    @(posedge clk); #1;
                    if (!full || in_ready || !dropped_event) begin
                        $fatal(1, "full queue did not assert backpressure and drop trap");
                    end

                    $finish;
                end
            endmodule
            `default_nettype wire
            """
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            iverilog,
            "-g2012",
            "-o",
            str(output),
            str(REPO_ROOT / "hdl" / "sc_aer_priority_queue.v"),
            str(testbench),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    sim = subprocess.run([vvp, str(output)], capture_output=True, text=True, check=False)
    assert sim.returncode == 0, sim.stderr + sim.stdout


def test_verilator_lints_queue_and_router_priority_interface() -> None:
    verilator = shutil.which("verilator")
    if verilator is None:
        pytest.skip("Verilator is required for router/queue interface lint")

    proc = subprocess.run(
        [
            verilator,
            "--lint-only",
            "-Wall",
            str(REPO_ROOT / "hdl" / "sc_aer_priority_queue.v"),
            str(REPO_ROOT / "hdl" / "sc_aer_router.v"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
