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
import os
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
    TRAP_CHECKSUM_MISMATCH,
    TRAP_INVALID_SELECTION,
    TRAP_PARTIAL_WRITE,
    TRAP_READ_ONLY_BANK,
    TRAP_STAGED_OVERFLOW,
    TRAP_STAGED_UNDERFLOW,
    TrapSpec,
    UPDATE_CHECKSUM_ALGORITHM,
)
from sc_neurocore.hdl_gen.bus_interface import generate_live_parameter_bank


ITERATIONS = 20_000
REPEATS = 7
OUTPUT = Path("benchmarks/results/local_python_2026-06-04_live_control_updates.json")
BENCHMARK_CPUSET = "8-9"


def live_control_spec(bus_protocol: str = "axi4_lite") -> MMIOUpdateSpec:
    """Return the deterministic Q8.8/Q16.16 live-control benchmark contract."""

    return MMIOUpdateSpec(
        bus_protocol=bus_protocol,
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
            ParameterBankSpec(
                bank_name="calibration",
                start_address_bytes=0x4000,
                parameter_count=1,
                parameter_names=("cal0",),
                q_format="Q8.8",
                writable=False,
            ),
        ),
        trap=TrapSpec(max_flags=2),
    )


def time_update_sequences(spec: MMIOUpdateSpec) -> tuple[float, int]:
    """Return nanoseconds per staged update-sequence build and checksum guard."""

    checksum = 0
    weight_count = spec.bank_by_name("weights").parameter_count
    start_ns = time.perf_counter_ns()
    for index in range(ITERATIONS):
        writes = spec.build_update_sequence("weights", index % weight_count, index & 0x7FFF)
        guard = next(write.value for write in writes if write.purpose == "write_checksum")
        checksum = (checksum + len(writes) + guard + writes[-1].value) & 0xFFFF_FFFF
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum


def time_static_regeneration(spec: MMIOUpdateSpec, module_name: str) -> tuple[float, int]:
    """Return nanoseconds per generated RTL regeneration and source-size checksum."""

    checksum = 0
    start_ns = time.perf_counter_ns()
    for _ in range(REPEATS):
        source = generate_live_parameter_bank(spec, module_name=module_name)
        checksum ^= len(source)
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / REPEATS, checksum


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _self_cgroup_path() -> str:
    cgroup = _read_text(Path("/proc/self/cgroup"))
    for line in cgroup.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            return parts[2].strip("/")
    return ""


def _cgroup_effective_cpuset() -> str:
    relative = _self_cgroup_path()
    if not relative:
        return ""
    return _read_text(Path("/sys/fs/cgroup") / relative / "cpuset.cpus.effective")


def _kernel_isolated_cpus() -> str:
    return _read_text(Path("/sys/devices/system/cpu/isolated"))


def _parse_cpuset(cpuset: str) -> list[int]:
    cpus: list[int] = []
    for raw_part in cpuset.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            cpus.extend(range(int(start_raw), int(end_raw) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def live_control_measurement_context(load_average_before: list[float] | None) -> dict[str, object]:
    """Return benchmark context including runtime cpuset evidence."""

    context = measurement_context(load_average_before)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    cgroup_effective_cpuset = _cgroup_effective_cpuset()
    kernel_isolated_cpus = _kernel_isolated_cpus()
    context.update(
        {
            "cgroup_path": _self_cgroup_path(),
            "cgroup_effective_cpuset": cgroup_effective_cpuset,
            "benchmark_requested_cpuset": BENCHMARK_CPUSET,
            "benchmark_requested_cpu_list": _parse_cpuset(BENCHMARK_CPUSET),
            "benchmark_affinity_matches_requested": affinity == _parse_cpuset(BENCHMARK_CPUSET),
            "runtime_cpuset_shield_claimed": cgroup_effective_cpuset == BENCHMARK_CPUSET
            or affinity == _parse_cpuset(BENCHMARK_CPUSET),
            "kernel_isolated_core_claimed": bool(kernel_isolated_cpus),
            "kernel_isolated_cpus": kernel_isolated_cpus,
        }
    )
    return context


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
    reg [5:0] external_trap_vector = 6'b000000;
    wire trap_latched;
    wire [5:0] trap_status_vector;
    wire staged_overflow;
    wire staged_underflow;
    wire update_pulse;
    wire apply_pulse;
    wire rollback_pulse;
    wire trap_clear_pulse;
    wire checksum_mismatch_pulse;
    wire invalid_selection_pulse;
    wire read_only_bank_pulse;
    wire partial_write_pulse;
    wire shadow_loaded;
    wire [2063:0] parameter_words;

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
        .checksum_mismatch_pulse(checksum_mismatch_pulse),
        .invalid_selection_pulse(invalid_selection_pulse),
        .read_only_bank_pulse(read_only_bank_pulse),
        .partial_write_pulse(partial_write_pulse),
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
        axi_write(32'h120, 32'h27E798F0);
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
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $finish(3);
        end

        axi_write(32'h110, 32'h00007FFF);
        axi_write(32'h114, 32'hFFFFFFFF);
        axi_write(32'h120, 32'h0E6BCD92);
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


def run_pcie_commit_sim(spec: MMIOUpdateSpec) -> dict[str, object]:
    """Compile and execute the PCIe-MMIO register-window commit contract."""

    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise RuntimeError("iverilog and vvp are required for PCIe MMIO live-control evidence")

    testbench = """
module tb_sc_live_pcie_params;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [31:0] write_addr = 32'd0;
    reg [31:0] write_data = 32'd0;
    reg [3:0] write_strobe = 4'hF;
    reg write_valid = 1'b0;
    wire write_ready;
    wire write_response_valid;
    wire write_error;
    reg [31:0] read_addr = 32'd0;
    reg read_valid = 1'b0;
    wire read_ready;
    wire [31:0] read_data;
    wire read_data_valid;
    wire read_error;
    reg [5:0] external_trap_vector = 6'b000000;
    wire trap_latched;
    wire [5:0] trap_status_vector;
    wire staged_overflow;
    wire staged_underflow;
    reg observed_checksum_mismatch = 1'b0;
    reg observed_invalid_selection = 1'b0;
    reg observed_read_only_bank = 1'b0;
    reg observed_partial_write = 1'b0;
    wire update_pulse;
    wire apply_pulse;
    wire rollback_pulse;
    wire trap_clear_pulse;
    wire checksum_mismatch_pulse;
    wire invalid_selection_pulse;
    wire read_only_bank_pulse;
    wire partial_write_pulse;
    wire shadow_loaded;
    wire [2063:0] parameter_words;

    always #5 clk = ~clk;
    always @(posedge clk) begin
        if (checksum_mismatch_pulse) begin
            observed_checksum_mismatch <= 1'b1;
        end
        if (invalid_selection_pulse) begin
            observed_invalid_selection <= 1'b1;
        end
        if (read_only_bank_pulse) begin
            observed_read_only_bank <= 1'b1;
        end
        if (partial_write_pulse) begin
            observed_partial_write <= 1'b1;
        end
    end

    sc_live_pcie_params dut (
        .pcie_clk(clk),
        .pcie_resetn(rst_n),
        .pcie_mmio_write_addr(write_addr),
        .pcie_mmio_write_data(write_data),
        .pcie_mmio_write_strobe(write_strobe),
        .pcie_mmio_write_valid(write_valid),
        .pcie_mmio_write_ready(write_ready),
        .pcie_mmio_write_response_valid(write_response_valid),
        .pcie_mmio_write_error(write_error),
        .pcie_mmio_read_addr(read_addr),
        .pcie_mmio_read_valid(read_valid),
        .pcie_mmio_read_ready(read_ready),
        .pcie_mmio_read_data(read_data),
        .pcie_mmio_read_data_valid(read_data_valid),
        .pcie_mmio_read_error(read_error),
        .trap_vector(external_trap_vector),
        .trap_latched(trap_latched),
        .trap_status_vector(trap_status_vector),
        .staged_overflow(staged_overflow),
        .staged_underflow(staged_underflow),
        .update_pulse(update_pulse),
        .apply_pulse(apply_pulse),
        .rollback_pulse(rollback_pulse),
        .trap_clear_pulse(trap_clear_pulse),
        .checksum_mismatch_pulse(checksum_mismatch_pulse),
        .invalid_selection_pulse(invalid_selection_pulse),
        .read_only_bank_pulse(read_only_bank_pulse),
        .partial_write_pulse(partial_write_pulse),
        .shadow_loaded(shadow_loaded),
        .parameter_words(parameter_words)
    );

    task pcie_write;
        input [31:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            write_addr = addr;
            write_data = data;
            write_valid = 1'b1;
            @(negedge clk);
            write_valid = 1'b0;
            @(negedge clk);
            if (write_error !== 1'b0) begin
                $finish(1);
            end
        end
    endtask

    task pcie_partial_write;
        input [31:0] addr;
        input [31:0] data;
        input [3:0] strobe;
        begin
            @(negedge clk);
            write_addr = addr;
            write_data = data;
            write_strobe = strobe;
            write_valid = 1'b1;
            @(negedge clk);
            write_valid = 1'b0;
            @(negedge clk);
            write_strobe = 4'hF;
            if (write_error !== 1'b1) begin
                $finish(14);
            end
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        pcie_partial_write(32'h110, 32'hFFFFEEEE, 4'h1);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words[15:0] !== 16'h0000) begin
            $finish(15);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[5] !== 1'b1 || observed_partial_write !== 1'b1) begin
            $finish(16);
        end

        pcie_write(32'h11C, 32'h0000003F);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $finish(17);
        end

        pcie_write(32'h108, 32'd0);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h110, 32'h00001234);
        pcie_write(32'h114, 32'h00000000);
        pcie_write(32'h120, 32'h00001234);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words[15:0] !== 16'h0000) begin
            $finish(5);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[2] !== 1'b1 || observed_checksum_mismatch !== 1'b1) begin
            $finish(6);
        end

        pcie_write(32'h11C, 32'h0000003F);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $finish(7);
        end

        pcie_write(32'h108, 32'd3);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h120, 32'h34B52FC7);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words[15:0] !== 16'h0000) begin
            $finish(8);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[3] !== 1'b1 || observed_invalid_selection !== 1'b1) begin
            $finish(9);
        end

        pcie_write(32'h11C, 32'h0000003F);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $finish(10);
        end

        pcie_write(32'h108, 32'd2);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h120, 32'h9ADDBE56);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words[2063:2048] !== 16'h0000) begin
            $finish(11);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[4] !== 1'b1 || observed_read_only_bank !== 1'b1) begin
            $finish(12);
        end

        pcie_write(32'h11C, 32'h0000003F);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $finish(13);
        end

        pcie_write(32'h108, 32'd0);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h120, 32'h1D7D9B35);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b1 || parameter_words[15:0] !== 16'h0000) begin
            $finish(2);
        end

        pcie_write(32'h108, 32'd2);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h100, 32'h00000002);
        repeat (2) @(negedge clk);
        if (parameter_words[15:0] !== 16'h1234 || parameter_words[2063:2048] !== 16'h0000 || shadow_loaded !== 1'b0) begin
            $finish(3);
        end
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000 || staged_overflow !== 1'b0 || staged_underflow !== 1'b0) begin
            $finish(4);
        end

        $finish(0);
    end
endmodule
"""

    start_ns = time.perf_counter_ns()
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        source_path = tmp / "sc_live_pcie_params.sv"
        tb_path = tmp / "tb_sc_live_pcie_params.sv"
        sim_path = tmp / "sc_live_pcie_params.out"
        source_path.write_text(
            generate_live_parameter_bank(spec, module_name="sc_live_pcie_params"),
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
        "checksum_mismatch_rejected": True,
        "checksum_mismatch_trap_bit": TRAP_CHECKSUM_MISMATCH,
        "invalid_selection_rejected": True,
        "invalid_selection_trap_bit": TRAP_INVALID_SELECTION,
        "read_only_bank_rejected": True,
        "read_only_bank_trap_bit": TRAP_READ_ONLY_BANK,
        "partial_write_rejected": True,
        "partial_write_trap_bit": TRAP_PARTIAL_WRITE,
        "retargeted_commit_preserved_shadow_identity": True,
        "passed": True,
    }


def main() -> int:
    load_average_before = load_average()
    spec = live_control_spec("axi4_lite")
    pcie_spec = live_control_spec("pcie")
    update_results = [time_update_sequences(spec) for _ in range(REPEATS)]
    pcie_update_results = [time_update_sequences(pcie_spec) for _ in range(REPEATS)]
    regeneration_results = [
        time_static_regeneration(spec, "sc_live_params") for _ in range(REPEATS)
    ]
    pcie_regeneration_results = [
        time_static_regeneration(pcie_spec, "sc_live_pcie_params") for _ in range(REPEATS)
    ]
    trap_capture = run_trap_capture_sim(spec)
    pcie_commit_capture = run_pcie_commit_sim(pcie_spec)
    update_ns = [float(item[0]) for item in update_results]
    pcie_update_ns = [float(item[0]) for item in pcie_update_results]
    regeneration_ns = [float(item[0]) for item in regeneration_results]
    pcie_regeneration_ns = [float(item[0]) for item in pcie_regeneration_results]

    report = {
        "benchmark": "live_control_update_and_trap_evidence",
        "benchmark_isolation_mode": "process-affinity-cpuset",
        "language": "Python+SystemVerilog",
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": (
            f"taskset -c {BENCHMARK_CPUSET} env PYTHONPATH=src "
            ".venv/bin/python benchmarks/bench_live_control_updates.py"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "measurement_context": live_control_measurement_context(load_average_before),
        "iterations": ITERATIONS,
        "repeats": REPEATS,
        "bus_protocols": {"axi4_lite": True, "pcie": True},
        "checksum_algorithm": UPDATE_CHECKSUM_ALGORITHM,
        "checksum_guard_samples": {
            "weights_w0_0x1234": spec.update_checksum("weights", 0, 0x1234),
            "weights_w0_staged_overflow_raw_words": 0x27E798F0,
            "weights_w0_staged_underflow_raw_words": 0x0E6BCD92,
        },
        "banks": [bank.to_dict() for bank in spec.banks],
        "control_registers": spec.control_register_addresses,
        "trap_bits": spec.trap_bits,
        "expected_trap_bits": {
            "staged_overflow": TRAP_STAGED_OVERFLOW,
            "staged_underflow": TRAP_STAGED_UNDERFLOW,
            "checksum_mismatch": TRAP_CHECKSUM_MISMATCH,
            "invalid_selection": TRAP_INVALID_SELECTION,
            "read_only_bank": TRAP_READ_ONLY_BANK,
            "partial_write": TRAP_PARTIAL_WRITE,
        },
        "valid_update_write_count": len(spec.build_update_sequence("weights", 0, 0x1234)),
        "live_update_sequence_median_ns": statistics.median(update_ns),
        "live_update_sequence_min_ns": min(update_ns),
        "live_update_sequence_max_ns": max(update_ns),
        "pcie_mmio_valid_update_write_count": len(
            pcie_spec.build_update_sequence("weights", 0, 0x1234)
        ),
        "pcie_mmio_live_update_sequence_median_ns": statistics.median(pcie_update_ns),
        "pcie_mmio_live_update_sequence_min_ns": min(pcie_update_ns),
        "pcie_mmio_live_update_sequence_max_ns": max(pcie_update_ns),
        "static_regeneration_median_ns": statistics.median(regeneration_ns),
        "static_regeneration_min_ns": min(regeneration_ns),
        "static_regeneration_max_ns": max(regeneration_ns),
        "pcie_mmio_static_regeneration_median_ns": statistics.median(pcie_regeneration_ns),
        "pcie_mmio_static_regeneration_min_ns": min(pcie_regeneration_ns),
        "pcie_mmio_static_regeneration_max_ns": max(pcie_regeneration_ns),
        "trap_capture": trap_capture,
        "pcie_mmio_commit_capture": pcie_commit_capture,
        "update_results": [
            {"ns_per_sequence": item[0], "checksum": item[1]} for item in update_results
        ],
        "pcie_mmio_update_results": [
            {"ns_per_sequence": item[0], "checksum": item[1]} for item in pcie_update_results
        ],
        "static_regeneration_results": [
            {"ns_per_regeneration": item[0], "checksum": item[1]} for item in regeneration_results
        ],
        "pcie_mmio_static_regeneration_results": [
            {"ns_per_regeneration": item[0], "checksum": item[1]}
            for item in pcie_regeneration_results
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
