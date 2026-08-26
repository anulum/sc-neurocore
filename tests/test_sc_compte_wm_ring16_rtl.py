# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Bit-exact co-simulation for the SC Compte 16-bin ring representative."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from sc_neurocore.network.sc_compte_wm import SCCompteWMNetworkSpec
from tests.toolchain_support import require_executable

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_compte_wm_ring16.v"
IVERILOG = require_executable("iverilog")
VVP = require_executable("vvp")
RECEIPT = ROOT / "benchmarks/results/bench_sc_compte_wm_ring16.json"
SCALE = 1 << 16
N_BINS = 16


def _quantized_weights() -> list[int]:
    """Derive the enrolled LUT from the live 16-target software footprint."""
    spec = SCCompteWMNetworkSpec()
    targets = np.arange(N_BINS, dtype=np.float64) * (360.0 / N_BINS)
    weights = spec.connectivity_footprint("ee", 0.0, targets)
    return [int(value) for value in np.rint(weights * SCALE).astype(np.int64)]


def _dense_oracle(gates: list[int], target: int, weights: list[int]) -> int:
    """Compute the independent unsigned Q16.16 no-autapse dense aggregate."""
    accumulator = 0
    for source, gate in enumerate(gates):
        if source != target:
            accumulator += gate * weights[(target - source) % N_BINS]
    return accumulator >> 16


def _literal_vector(values: list[int]) -> str:
    return "\n".join(
        f"        load_gate(4'd{index}, 32'd{value});" for index, value in enumerate(values)
    )


def _run_vector_statements(gates: list[int], targets: tuple[int, ...]) -> str:
    weights = _quantized_weights()
    checks = "\n".join(
        f"        run_target(4'd{target}, 32'd{_dense_oracle(gates, target, weights)});"
        for target in targets
    )
    return f"{_literal_vector(gates)}\n{checks}"


def test_frozen_lut_is_live_source_derived_and_schema_bounded() -> None:
    weights = _quantized_weights()
    assert weights == [
        106168,
        80982,
        61755,
        59755,
        59714,
        59714,
        59714,
        59714,
        59714,
        59714,
        59714,
        59714,
        59714,
        59755,
        61755,
        80982,
    ]
    assert sum(weights) == 1_048_578
    assert sum(weights) - weights[0] == 942_410

    schema = json.loads(
        (ROOT / "src/sc_neurocore/network/schemas/sc_compte_wm_network.json").read_text()
    )
    boundary = schema["hardware_boundary"]
    assert boundary["representative_bins"] == N_BINS
    assert boundary["fixed_point_format"] == "unsigned Q16.16"
    assert boundary["latency_cycles"] == N_BINS


def test_iverilog_matches_independent_dense_oracle(tmp_path: Path) -> None:
    zero = [0] * N_BINS
    unity = [SCALE] * N_BINS
    ramp = [index * 4096 for index in range(N_BINS)]
    mixed = [
        65536,
        0,
        8192,
        32768,
        16384,
        49152,
        1024,
        64512,
        22222,
        44444,
        5555,
        33333,
        12345,
        54321,
        7777,
        60000,
    ]
    vectors = "\n".join(
        (
            _run_vector_statements(zero, (0, 7, 15)),
            _run_vector_statements(unity, tuple(range(N_BINS))),
            _run_vector_statements(ramp, (0, 1, 5, 9, 15)),
            _run_vector_statements(mixed, (0, 3, 8, 12, 15)),
        )
    )
    testbench = f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg load_valid = 1'b0;
    reg [3:0] load_index = 4'd0;
    reg [31:0] load_gate_q1616 = 32'd0;
    reg start = 1'b0;
    reg [3:0] target_bin = 4'd0;
    wire busy;
    wire done;
    wire [31:0] aggregate_q1616;
    integer cycles;

    always #5 clk = ~clk;
    sc_compte_wm_ring16 dut (
        .clk(clk), .rst_n(rst_n), .load_valid(load_valid),
        .load_index(load_index), .load_gate_q1616(load_gate_q1616),
        .start(start), .target_bin(target_bin), .busy(busy), .done(done),
        .aggregate_q1616(aggregate_q1616)
    );

    task load_gate(input [3:0] index, input [31:0] value);
        begin
            @(negedge clk);
            load_valid = 1'b1;
            load_index = index;
            load_gate_q1616 = value;
            @(posedge clk);
            @(negedge clk);
            load_valid = 1'b0;
        end
    endtask

    task run_target(input [3:0] target, input [31:0] expected);
        begin
            @(negedge clk);
            target_bin = target;
            start = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start = 1'b0;
            cycles = 0;
            while (!done) begin
                @(negedge clk);
                cycles = cycles + 1;
                if (cycles == 4) begin
                    load_valid = 1'b1;
                    load_index = 4'd15;
                    load_gate_q1616 = 32'd123;
                end else if (cycles == 5) begin
                    load_valid = 1'b0;
                end
                if (cycles > 16)
                    $fatal(1, "done missed 16-cycle contract");
            end
            load_valid = 1'b0;
            if (cycles != 16)
                $fatal(1, "latency got %0d want 16", cycles);
            if (aggregate_q1616 !== expected)
                $fatal(1, "target %0d got %0d want %0d", target,
                       aggregate_q1616, expected);
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
{vectors}
        $display("PASS sc_compte_wm_ring16");
        $finish(0);
    end
endmodule
"""
    tb_path = tmp_path / "sc_compte_wm_ring16_tb.v"
    sim_path = tmp_path / "sc_compte_wm_ring16.out"
    tb_path.write_text(testbench, encoding="utf-8")
    compile_result = subprocess.run(
        [str(IVERILOG), "-g2012", "-o", str(sim_path), str(RTL), str(tb_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    sim_result = subprocess.run(
        [str(VVP), str(sim_path)], capture_output=True, text=True, check=False
    )
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr
    assert "PASS sc_compte_wm_ring16" in sim_result.stdout


def test_committed_receipt_is_source_custodied_and_honest() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["identity"] == "SC-COMPTE-WM-NETWORK-RING16-RTL"
    assert receipt["cosimulation"]["passed"] is True
    assert receipt["formal"]["passed"] is True
    assert receipt["formal"]["datapath_formally_claimed"] is False
    assert receipt["post_optimization_equivalence"]["passed"] is True
    assert receipt["synthesis"]["passed"] is True
    assert receipt["claim_boundary"]["physical_device_evidence"] is False
    for relative, expected in receipt["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
