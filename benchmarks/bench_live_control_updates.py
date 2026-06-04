# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live-control update and trap evidence artefact writer

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import time

from _benchmark_context import load_average, measurement_context
from sc_neurocore.compiler.live_control import (
    MMIOUpdateSpec,
    ParameterBankSpec,
    TRAP_STAGED_OVERFLOW,
    TRAP_STAGED_UNDERFLOW,
    TrapSpec,
)
from sc_neurocore.hdl_gen.bus_interface import generate_live_parameter_bank


ITERATIONS = 20_000
REPEATS = 7
OUTPUT = Path("benchmarks/results/local_python_2026-06-04_live_control_updates.json")


def live_control_spec() -> MMIOUpdateSpec:
    """Return the deterministic Q8.8/Q16.16 live-control benchmark contract."""

    return MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        control_base_address_bytes=0x100,
        banks=(
            ParameterBankSpec(
                bank_name="weights",
                start_address_bytes=0x2000,
                parameter_count=64,
                parameter_names=tuple(f"w{i}" for i in range(64)),
                q_format="Q8.8",
            ),
            ParameterBankSpec(
                bank_name="kuramoto",
                start_address_bytes=0x3000,
                parameter_count=32,
                parameter_names=tuple(f"k{i}" for i in range(32)),
                q_format="Q16.16",
            ),
        ),
        trap=TrapSpec(max_flags=2),
    )


def time_update_sequences(spec: MMIOUpdateSpec) -> tuple[float, int]:
    """Return nanoseconds per staged update-sequence build and checksum guard."""

    checksum = 0
    start_ns = time.perf_counter_ns()
    for index in range(ITERATIONS):
        writes = spec.build_update_sequence("weights", index % 64, index & 0x7FFF)
        checksum ^= len(writes)
        checksum ^= writes[-2].value
        checksum ^= writes[-1].value
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum


def time_static_regeneration(spec: MMIOUpdateSpec) -> tuple[float, int]:
    """Return nanoseconds per generated RTL regeneration and source-size checksum."""

    checksum = 0
    start_ns = time.perf_counter_ns()
    for _ in range(REPEATS):
        source = generate_live_parameter_bank(spec, module_name="sc_live_params")
        checksum ^= len(source)
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / REPEATS, checksum


def run_trap_capture_sim(spec: MMIOUpdateSpec) -> dict[str, object]:
    """Compile and execute generated RTL that captures overflow and underflow traps."""

    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise RuntimeError("iverilog and vvp are required for live-control trap evidence")

    testbench = """
module tb_sc_live_params;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [31:0] awaddr = 32'd0;
    reg awvalid = 1'b0;
    wire awready;
    reg [31:0] wdata = 32'd0;
    reg [3:0] wstrb = 4'hF;
    reg wvalid = 1'b0;
    wire wready;
    wire [1:0] bresp;
    wire bvalid;
    reg bready = 1'b0;
    reg [31:0] araddr = 32'd0;
    reg arvalid = 1'b0;
    wire arready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire rvalid;
    reg rready = 1'b0;
    reg [1:0] external_trap_vector = 2'b00;
    wire trap_latched;
    wire [1:0] trap_status_vector;
    wire staged_overflow;
    wire staged_underflow;
    wire update_pulse;
    wire apply_pulse;
    wire rollback_pulse;
    wire trap_clear_pulse;
    wire shadow_loaded;
    wire [2047:0] parameter_words;

    always #5 clk = ~clk;

    sc_live_params dut (
        .S_AXI_ACLK(clk),
        .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(awaddr),
        .S_AXI_AWVALID(awvalid),
        .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata),
        .S_AXI_WSTRB(wstrb),
        .S_AXI_WVALID(wvalid),
        .S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp),
        .S_AXI_BVALID(bvalid),
        .S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr),
        .S_AXI_ARVALID(arvalid),
        .S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata),
        .S_AXI_RRESP(rresp),
        .S_AXI_RVALID(rvalid),
        .S_AXI_RREADY(rready),
        .trap_vector(external_trap_vector),
        .trap_latched(trap_latched),
        .trap_status_vector(trap_status_vector),
        .staged_overflow(staged_overflow),
        .staged_underflow(staged_underflow),
        .update_pulse(update_pulse),
        .apply_pulse(apply_pulse),
        .rollback_pulse(rollback_pulse),
        .trap_clear_pulse(trap_clear_pulse),
        .shadow_loaded(shadow_loaded),
        .parameter_words(parameter_words)
    );

    task axi_write;
        input [31:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            awaddr = addr;
            wdata = data;
            awvalid = 1'b1;
            wvalid = 1'b1;
            bready = 1'b1;
            @(negedge clk);
            awvalid = 1'b0;
            wvalid = 1'b0;
            @(negedge clk);
            bready = 1'b0;
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        axi_write(32'h108, 32'd0);
        axi_write(32'h10C, 32'd0);
        axi_write(32'h110, 32'h00010000);
        axi_write(32'h114, 32'h00000000);
        axi_write(32'h120, 32'h00010000);
        axi_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b1 || trap_status_vector[0] !== 1'b1 || staged_overflow !== 1'b1) begin
            $finish(1);
        end
        if (shadow_loaded !== 1'b0 || parameter_words[15:0] !== 16'h0000) begin
            $finish(2);
        end

        axi_write(32'h11C, 32'h00000003);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 2'b00) begin
            $finish(3);
        end

        axi_write(32'h110, 32'h00007FFF);
        axi_write(32'h114, 32'hFFFFFFFF);
        axi_write(32'h120, 32'hFFFF8000);
        axi_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b1 || trap_status_vector[1] !== 1'b1 || staged_underflow !== 1'b1) begin
            $finish(4);
        end

        $finish(0);
    end
endmodule
"""

    start_ns = time.perf_counter_ns()
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        source_path = tmp / "sc_live_params.sv"
        tb_path = tmp / "tb_sc_live_params.sv"
        sim_path = tmp / "sc_live_params.out"
        source_path.write_text(
            generate_live_parameter_bank(spec, module_name="sc_live_params"),
            encoding="utf-8",
        )
        tb_path.write_text(testbench, encoding="utf-8")
        compile_result = subprocess.run(
            [iverilog, "-g2012", "-o", str(sim_path), str(source_path), str(tb_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(compile_result.stderr)
        compile_done_ns = time.perf_counter_ns()
        run_result = subprocess.run(
            [vvp, str(sim_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if run_result.returncode != 0:
            raise RuntimeError(run_result.stdout + run_result.stderr)
    end_ns = time.perf_counter_ns()
    return {
        "iverilog": iverilog,
        "vvp": vvp,
        "compile_ns": compile_done_ns - start_ns,
        "simulation_ns": end_ns - compile_done_ns,
        "passed": True,
    }


def main() -> int:
    load_average_before = load_average()
    spec = live_control_spec()
    update_results = [time_update_sequences(spec) for _ in range(REPEATS)]
    regeneration_results = [time_static_regeneration(spec) for _ in range(REPEATS)]
    trap_capture = run_trap_capture_sim(spec)
    update_ns = [float(item[0]) for item in update_results]
    regeneration_ns = [float(item[0]) for item in regeneration_results]

    report = {
        "benchmark": "live_control_update_and_trap_evidence",
        "language": "Python+SystemVerilog",
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": (
            "taskset -c 10-11 env PYTHONPATH=src "
            ".venv/bin/python benchmarks/bench_live_control_updates.py"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "measurement_context": measurement_context(load_average_before),
        "iterations": ITERATIONS,
        "repeats": REPEATS,
        "banks": [bank.to_dict() for bank in spec.banks],
        "control_registers": spec.control_register_addresses,
        "trap_bits": spec.trap_bits,
        "expected_trap_bits": {
            "staged_overflow": TRAP_STAGED_OVERFLOW,
            "staged_underflow": TRAP_STAGED_UNDERFLOW,
        },
        "valid_update_write_count": len(spec.build_update_sequence("weights", 0, 0x1234)),
        "live_update_sequence_median_ns": statistics.median(update_ns),
        "live_update_sequence_min_ns": min(update_ns),
        "live_update_sequence_max_ns": max(update_ns),
        "static_regeneration_median_ns": statistics.median(regeneration_ns),
        "static_regeneration_min_ns": min(regeneration_ns),
        "static_regeneration_max_ns": max(regeneration_ns),
        "trap_capture": trap_capture,
        "update_results": [
            {"ns_per_sequence": item[0], "checksum": item[1]} for item in update_results
        ],
        "static_regeneration_results": [
            {"ns_per_regeneration": item[0], "checksum": item[1]} for item in regeneration_results
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
