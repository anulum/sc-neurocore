# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for research-stage Kuramoto HDL emission

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, cast

import pytest

from sc_neurocore.hdl_gen import KuramotoEmitter
from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator


def test_kuramoto_emitter_has_expected_ports_and_helpers() -> None:
    emitter = KuramotoEmitter(
        module_name="kuramoto_top",
        n_oscillators=3,
        omegas=[0.9, 1.0, 1.1],
        initial_phases=[0.0, 0.2, 0.4],
    )
    code = emitter.generate()
    assert "module kuramoto_top" in code
    assert "input wire step_en" in code
    assert "output reg update_done" in code
    assert "function automatic signed [DATA_WIDTH-1:0] sin_lut;" in code
    assert "wire signed [DATA_WIDTH-1:0] phase_diff_0_1" in code
    assert "assign phase_bus[71:48] = phase_reg_2;" in code


def test_kuramoto_emitter_rejects_configuration_mismatch() -> None:
    try:
        KuramotoEmitter(
            n_oscillators=3,
            omegas=[1.0, 1.1],
            initial_phases=[0.0, 0.1, 0.2],
        )
    except ValueError as exc:
        assert "omegas length must equal n_oscillators" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for omega length mismatch")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_oscillators": 0}, "n_oscillators must be >= 1"),
        ({"data_width": 15}, "data_width must be >= 16"),
        ({"fraction": 0}, "fraction must satisfy 0 < fraction < data_width"),
        ({"fraction": 24}, "fraction must satisfy 0 < fraction < data_width"),
        ({"lut_size": 8}, "lut_size must be a power of two >= 16"),
        ({"lut_size": 48}, "lut_size must be a power of two >= 16"),
        (
            {"n_oscillators": 2, "initial_phases": [0.0]},
            "initial_phases length must equal n_oscillators",
        ),
    ],
)
def test_kuramoto_emitter_rejects_invalid_structural_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KuramotoEmitter(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.0}, "dt must be finite and positive"),
        ({"dt": float("nan")}, "dt must be finite and positive"),
        ({"coupling": float("inf")}, "coupling must be finite"),
        ({"omegas": [1.0, float("nan")]}, "omegas must contain only finite values"),
        (
            {"initial_phases": [0.0, float("-inf")]},
            "initial_phases must contain only finite values",
        ),
    ],
)
def test_kuramoto_emitter_rejects_invalid_numerical_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    base_kwargs: dict[str, Any] = {
        "n_oscillators": 2,
        "omegas": [1.0, 1.1],
        "initial_phases": [0.0, 0.2],
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        KuramotoEmitter(**base_kwargs)


def test_kuramoto_emitter_rejects_fixed_point_format_that_cannot_hold_phase_modulus() -> None:
    with pytest.raises(ValueError, match="fixed-point format cannot represent 2pi"):
        KuramotoEmitter(data_width=16, fraction=15)


def test_kuramoto_emitter_rejects_configuration_that_requires_multi_wrap_step() -> None:
    with pytest.raises(ValueError, match="single-step phase advance must stay below 2pi"):
        KuramotoEmitter(n_oscillators=1, omegas=[100.0], dt=0.1)


def test_kuramoto_emitter_fixed_point_reference_matches_known_coupled_step() -> None:
    emitter = KuramotoEmitter(
        n_oscillators=2,
        omegas=[0.0, 0.0],
        initial_phases=[0.0, 1.5707963267948966],
        coupling=1.0,
        dt=0.01,
        data_width=24,
        fraction=16,
        lut_size=64,
    )

    assert emitter.initial_phase_state_fixed() == [0, 102944]
    assert emitter.fixed_point_step(emitter.initial_phase_state_fixed()) == [327, 102618]


@pytest.mark.parametrize(
    ("phase_state", "message"),
    [
        ([0], "phase_state length must equal n_oscillators"),
        ([True, 0], "phase_state entries must be integers"),
        ([0.25, 0], "phase_state entries must be integers"),
        ([-1, 0], "phase_state entries must satisfy 0 <= phase < phase modulus"),
        (
            [int(round(2.0 * math.pi * (1 << 16))), 0],
            "phase_state entries must satisfy 0 <= phase < phase modulus",
        ),
    ],
)
def test_kuramoto_emitter_fixed_point_step_rejects_noncanonical_phase_state(
    phase_state: list[object], message: str
) -> None:
    emitter = KuramotoEmitter(n_oscillators=2, fraction=16)

    with pytest.raises(ValueError, match=message):
        emitter.fixed_point_step(cast(list[int], phase_state))


@pytest.mark.parametrize(
    ("phase_state", "message"),
    [
        ([0], "phase_state length must equal n_oscillators"),
        ([False, 0], "phase_state entries must be integers"),
        ([1.5, 0], "phase_state entries must be integers"),
        ([-1, 0], "phase_state entries must satisfy 0 <= phase < phase modulus"),
        (
            [int(round(2.0 * math.pi * (1 << 16))), 0],
            "phase_state entries must satisfy 0 <= phase < phase modulus",
        ),
    ],
)
def test_kuramoto_emitter_fixed_state_to_float_rejects_noncanonical_phase_state(
    phase_state: list[object], message: str
) -> None:
    emitter = KuramotoEmitter(n_oscillators=2, fraction=16)

    with pytest.raises(ValueError, match=message):
        emitter.fixed_state_to_float(cast(list[int], phase_state))


def test_kuramoto_emitter_characterises_fixed_point_error_against_float_reference() -> None:
    emitter = KuramotoEmitter(
        n_oscillators=4,
        omegas=[0.88, 0.96, 1.04, 1.12],
        initial_phases=[0.05, 0.9, 1.7, 2.6],
        coupling=1.0,
        dt=2.5e-4,
        data_width=24,
        fraction=16,
        lut_size=128,
    )

    summary = emitter.fixed_point_error_summary(steps=32)

    assert summary["steps"] == 32
    assert summary["oscillator_count"] == 4
    assert cast(float, summary["max_abs_phase_error_rad"]) < 0.01
    assert cast(float, summary["rms_phase_error_rad"]) < 0.005
    assert summary["final_fixed_phases_rad"] != summary["final_float_phases_rad"]


def test_kuramoto_emitter_fixed_point_error_summary_requires_positive_steps() -> None:
    emitter = KuramotoEmitter()

    with pytest.raises(ValueError, match="steps must be >= 1"):
        emitter.fixed_point_error_summary(steps=0)


def test_kuramoto_rtl_error_report_cli_writes_deterministic_gate(tmp_path: Path) -> None:
    output = tmp_path / "kuramoto_rtl_fixed_point_error.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_kuramoto_rtl_error_report.py",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["report_type"] == "kuramoto_rtl_fixed_point_error"
    assert payload["hardware_measurement_claimed"] is False
    assert payload["failed_cases"] == 0
    assert payload["passed_cases"] == payload["total_cases"]
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert {case["case_name"] for case in payload["cases"]} >= {
        "higher_coupling_quartet_short",
        "low_coupling_quartet_short",
        "no_coupling_single_oscillator",
        "wrap_boundary_pair",
        "near_antiphase_pair",
    }
    assert all(case["config"]["phase_regime"] for case in payload["cases"])

    higher_coupling = next(
        case for case in payload["cases"] if case["case_name"] == "higher_coupling_quartet_short"
    )
    assert higher_coupling["passed"] is True
    assert higher_coupling["rust_solver_parity"]["available"] is True
    assert (
        higher_coupling["rust_solver_parity"]["max_abs_phase_error_rad"]
        < higher_coupling["thresholds"]["rust_max_abs_phase_error_rad"]
    )
    assert (
        higher_coupling["rust_solver_parity"]["order_parameter_error"]
        < higher_coupling["thresholds"]["rust_order_parameter_error"]
    )
    assert higher_coupling["config"]["oscillator_count"] == 4
    assert higher_coupling["config"]["coupling"] == 1.0
    assert (
        higher_coupling["summary"]["max_abs_phase_error_rad"]
        < higher_coupling["thresholds"]["max_abs_phase_error_rad"]
    )
    assert (
        higher_coupling["summary"]["rms_phase_error_rad"]
        < higher_coupling["thresholds"]["rms_phase_error_rad"]
    )


def test_committed_kuramoto_rtl_error_report_matches_reference() -> None:
    report_path = Path("benchmarks/results/kuramoto_rtl_fixed_point_error.json")

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["failed_cases"] == 0
    assert payload["total_cases"] >= 5
    for case in payload["cases"]:
        config = case["config"]
        assert config["phase_regime"] in {
            "nominal_higher_coupling",
            "nominal_low_coupling",
            "single_oscillator_no_coupling",
            "phase_modulus_wrap_boundary",
            "near_antiphase_circular_error",
        }
        emitter = KuramotoEmitter(
            n_oscillators=config["oscillator_count"],
            omegas=config["omegas"],
            initial_phases=config["initial_phases"],
            coupling=config["coupling"],
            dt=config["dt"],
            data_width=config["data_width"],
            fraction=config["fraction"],
            lut_size=config["lut_size"],
        )
        assert case["summary"] == emitter.fixed_point_error_summary(steps=config["steps"])
        assert case["rust_solver_parity"]["available"] is True
        assert (
            case["rust_solver_parity"]["max_abs_phase_error_rad"]
            < case["thresholds"]["rust_max_abs_phase_error_rad"]
        )
        assert (
            case["rust_solver_parity"]["order_parameter_error"]
            < case["thresholds"]["rust_order_parameter_error"]
        )
        assert case["passed"] is True


def test_verilog_generator_can_emit_kuramoto_phase() -> None:
    gen = VerilogGenerator(module_name="kuramoto_wrap")
    code = gen.emit_kuramoto_phase(
        n_oscillators=2,
        omegas=[0.95, 1.05],
        initial_phases=[0.0, 0.1],
        coupling=0.05,
    )
    assert "module kuramoto_wrap" in code
    assert "localparam integer N_OSC = 2;" in code
    assert "wire signed [DATA_WIDTH-1:0] phase_velocity_0" in code


def test_kuramoto_emitter_smoke_compiles_with_iverilog(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    if iverilog is None:
        raise AssertionError("iverilog must be available for HDL smoke tests")

    emitter = KuramotoEmitter(
        module_name="kuramoto_compile",
        n_oscillators=4,
        omegas=[0.8, 1.0, 1.1, 1.3],
        initial_phases=[0.0, 0.3, 0.6, 0.9],
        coupling=0.12,
        dt=5e-3,
    )
    rtl_path = tmp_path / "kuramoto_compile.v"
    rtl_path.write_text(emitter.generate())

    result = subprocess.run(
        [iverilog, "-g2012", "-t", "null", str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_kuramoto_emitter_large_lut_compiles_without_index_truncation(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    if iverilog is None:
        raise AssertionError("iverilog must be available for HDL smoke tests")

    emitter = KuramotoEmitter(
        module_name="kuramoto_large_lut",
        n_oscillators=2,
        lut_size=512,
    )
    rtl_path = tmp_path / "kuramoto_large_lut.v"
    rtl_path.write_text(emitter.generate())

    result = subprocess.run(
        [iverilog, "-g2012", "-t", "null", str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Numeric constant truncated" not in result.stderr


def test_kuramoto_emitter_hdl_matches_no_coupling_fixed_point_step(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for HDL simulation tests")

    emitter = KuramotoEmitter(
        module_name="kuramoto_no_coupling_step",
        n_oscillators=1,
        omegas=[1.0],
        initial_phases=[0.0],
        coupling=0.0,
        dt=0.01,
        data_width=24,
        fraction=16,
    )
    expected_next = emitter.fixed_point_step(emitter.initial_phase_state_fixed())
    source = (
        emitter.generate()
        + f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg step_en = 1'b0;
    wire update_done;
    wire [23:0] phase_bus;

    kuramoto_no_coupling_step dut (
        .clk(clk),
        .rst_n(rst_n),
        .step_en(step_en),
        .update_done(update_done),
        .phase_bus(phase_bus)
    );

    always #5 clk = ~clk;

    initial begin
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        step_en = 1'b1;
        @(posedge clk);
        @(negedge clk);

        if (phase_bus !== 24'd{expected_next[0]}) begin
            $fatal(
                1,
                "expected one fixed-point phase step of {expected_next[0]}, observed %0d",
                phase_bus
            );
        end
        if (update_done !== 1'b1) begin
            $fatal(1, "expected update_done pulse after enabled step");
        end

        step_en = 1'b0;
        @(posedge clk);
        @(negedge clk);
        if (update_done !== 1'b0) begin
            $fatal(1, "expected update_done to clear when step_en is low");
        end

        $finish(0);
    end
endmodule
"""
    )
    rtl_path = tmp_path / "kuramoto_no_coupling_step.v"
    sim_path = tmp_path / "kuramoto_no_coupling_step.out"
    rtl_path.write_text(source)

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    sim_result = subprocess.run(
        [vvp, str(sim_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr


def test_kuramoto_emitter_hdl_matches_coupled_two_oscillator_fixed_point_step(
    tmp_path: Path,
) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for HDL simulation tests")

    emitter = KuramotoEmitter(
        module_name="kuramoto_coupled_step",
        n_oscillators=2,
        omegas=[0.0, 0.0],
        initial_phases=[0.0, 1.5707963267948966],
        coupling=1.0,
        dt=0.01,
        data_width=24,
        fraction=16,
        lut_size=64,
    )
    initial_state = emitter.initial_phase_state_fixed()
    expected_next = emitter.fixed_point_step(initial_state)
    source = (
        emitter.generate()
        + f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg step_en = 1'b0;
    wire update_done;
    wire [47:0] phase_bus;

    kuramoto_coupled_step dut (
        .clk(clk),
        .rst_n(rst_n),
        .step_en(step_en),
        .update_done(update_done),
        .phase_bus(phase_bus)
    );

    always #5 clk = ~clk;

    initial begin
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        if (phase_bus[23:0] !== 24'd{initial_state[0]} || phase_bus[47:24] !== 24'd{initial_state[1]}) begin
            $fatal(
                1,
                "unexpected initial phases p0=%0d p1=%0d",
                phase_bus[23:0],
                phase_bus[47:24]
            );
        end

        step_en = 1'b1;
        @(posedge clk);
        @(negedge clk);

        if (phase_bus[23:0] !== 24'd{expected_next[0]} || phase_bus[47:24] !== 24'd{expected_next[1]}) begin
            $fatal(
                1,
                "unexpected coupled phase step p0=%0d p1=%0d",
                phase_bus[23:0],
                phase_bus[47:24]
            );
        end
        if (update_done !== 1'b1) begin
            $fatal(1, "expected update_done pulse after coupled step");
        end

        $finish(0);
    end
endmodule
"""
    )
    rtl_path = tmp_path / "kuramoto_coupled_step.v"
    sim_path = tmp_path / "kuramoto_coupled_step.out"
    rtl_path.write_text(source)

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    sim_result = subprocess.run(
        [vvp, str(sim_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr
