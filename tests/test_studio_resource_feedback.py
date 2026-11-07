# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resource and timing feedback on a real FPGA flow

"""Judge emitted graphs by real Yosys and nextpnr-ice40 runs, and every verdict branch."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from typing import Any

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraph, ScGraphBuilder  # noqa: E402
from sc_neurocore.studio import resource_feedback as feedback  # noqa: E402
from sc_neurocore.studio import synthesis  # noqa: E402
from tests.test_ir_sv_target_report import (  # noqa: E402
    MEASUREMENTS,
    _dense,
    _dense_with_ports,
    _graph,
)


def _tools_reason() -> str | None:
    """Return why the real-flow tests cannot run here, or None when they can."""
    yosys, nextpnr = shutil.which("yosys"), shutil.which("nextpnr-ice40")
    if yosys is None or nextpnr is None:
        return "yosys and nextpnr-ice40 are not both on PATH"
    if feedback.CALIBRATION_TOOLCHAIN["yosys"] not in feedback._version(yosys, TOOL_TIMEOUT_S):
        return "the yosys on PATH is not the calibration's Yosys 0.70"
    return None


#: Per-tool wall-clock limit these tests give Yosys and nextpnr. A reference value for
#: the hosts that run them (hosted CI runners, the workstation); the library has no
#: default, and converting test limits to host-derived values is a tracked backlog item.
TOOL_TIMEOUT_S = 900.0

REASON = _tools_reason()
if REASON is not None and os.environ.get("SC_NEUROCORE_REQUIRE_FPGA_TOOLS") == "1":
    raise RuntimeError(f"the FPGA resource-check job requires its tools: {REASON}")
real_flow = pytest.mark.skipif(
    REASON is not None, reason=f"{REASON}; the resource-check CI job runs these"
)


class _LoweredEstimate:
    """A real graph whose emitted report claims a smaller bound than the engine's."""

    def __init__(self, graph: ScGraph, logic_cells: int) -> None:
        self._graph = graph
        self._logic_cells = logic_cells

    @property
    def name(self) -> str:
        return self._graph.name

    def emit_sv_with_report(self, target: str, clock_mhz: int) -> tuple[str, Mapping[str, Any]]:
        verilog, report = self._graph.emit_sv_with_report(target, clock_mhz)
        return verilog, {**report, "logic_cells_estimated": self._logic_cells}


def _verdicts(result: Mapping[str, Any]) -> dict[str, str]:
    return {check["name"]: check["verdict"] for check in result["checks"]}


def _calibrated(logic_cells: int = 100, path_ns: float = 50.0) -> dict[str, Any]:
    return {
        "static_estimate": True,
        "logic_cells_estimated": logic_cells,
        "critical_path_estimate_ns": path_ns,
    }


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        (["pass", "pass"], "pass"),
        (["pass", "violation", "undetermined"], "violation"),
        (["pass", "undetermined"], "undetermined"),
        ([], "undetermined"),
    ],
)
def test_overall_verdict_prefers_violation_then_undetermined(
    verdicts: list[str], expected: str
) -> None:
    """A violation wins; undetermined is never a pass; no check is undetermined."""
    checks = [{"name": f"c{i}", "verdict": v, "detail": ""} for i, v in enumerate(verdicts)]
    assert feedback.overall_verdict(checks) == expected


def test_io_fits_up_to_the_package_pins_and_one_more_violates() -> None:
    """39 port bits fit the SG48; 40 do not."""
    assert feedback.io_check(feedback.ICE40_UP5K_SG48_IO_PINS)["verdict"] == "pass"
    assert feedback.io_check(feedback.ICE40_UP5K_SG48_IO_PINS + 1)["verdict"] == "violation"


def test_a_core_above_the_device_violates_and_a_harness_overflow_is_undetermined() -> None:
    """Only the core's own size decides capacity; a fitting core in an overflowing harness is not judged."""
    cells = feedback.ICE40_UP5K_LOGIC_CELLS
    assert feedback.capacity_check(cells + 1, cells + 200)["verdict"] == "violation"
    fits = feedback.capacity_check(cells, cells + 200)
    assert fits["verdict"] == "undetermined" and "five-pin harness (5480)" in fits["detail"]


def test_timing_is_met_at_the_constraint_and_missed_just_below_or_on_any_clock() -> None:
    """Every clock must reach its constraint; no clocks is no verdict."""
    assert feedback.timing_met({"clk": {"achieved": 12.0, "constraint": 12.0}}) is True
    assert feedback.timing_met({"clk": {"achieved": 11.99, "constraint": 12.0}}) is False
    two = {"a": {"achieved": 30.0, "constraint": 12.0}, "b": {"achieved": 10.0, "constraint": 12.0}}
    assert feedback.timing_met(two) is False
    assert feedback.timing_met({}) is None


def test_estimate_check_boundary_is_inclusive_and_one_above_violates() -> None:
    """Equal to the bound passes, one cell or one path step above violates."""
    at_bound = {"clk": {"achieved": 20.0, "constraint": 12.0}}
    assert (
        feedback._estimate_check(_calibrated(100, 50.0), 100, at_bound, True)["verdict"] == "pass"
    )
    assert (
        feedback._estimate_check(_calibrated(100, 50.0), 101, at_bound, True)["verdict"]
        == "violation"
    )
    slower = {"clk": {"achieved": 19.9, "constraint": 12.0}}
    check = feedback._estimate_check(_calibrated(100, 50.0), 100, slower, True)
    assert check["verdict"] == "violation" and "critical path" in check["detail"]


def test_estimate_check_judges_the_slowest_clock() -> None:
    """Every clock counts: one slow clock exceeds the path bound."""
    clocks = {
        "a": {"achieved": 40.0, "constraint": 12.0},
        "b": {"achieved": 15.0, "constraint": 12.0},
    }
    assert (
        feedback._estimate_check(_calibrated(10_000, 50.0), 1, clocks, True)["verdict"]
        == "violation"
    )


def test_estimate_check_is_undetermined_on_a_toolchain_mismatch_or_missing_numbers() -> None:
    """Another toolchain or a missing routed size never passes the comparison."""
    clocks = {"clk": {"achieved": 30.0, "constraint": 12.0}}
    mismatch = feedback._estimate_check(_calibrated(), 10, clocks, False)
    assert mismatch == {
        "name": "estimate",
        "verdict": "undetermined",
        "detail": "calibration toolchain mismatch",
    }
    assert feedback._estimate_check(_calibrated(), None, clocks, True)["verdict"] == "undetermined"
    assert feedback._estimate_check(_calibrated(), 10, {}, True)["verdict"] == "undetermined"


def test_harness_shifts_every_data_bit_and_keeps_clock_and_reset() -> None:
    """Five pins drive all core inputs and observe all core outputs."""
    ports = {
        "clk": {"direction": "input", "bits": [2]},
        "rst_n": {"direction": "input", "bits": [3]},
        "a": {"direction": "input", "bits": [4, 5, 6]},
        "b": {"direction": "input", "bits": [7]},
        "y": {"direction": "output", "bits": [8, 9]},
    }
    harness, stub, input_bits, output_bits = feedback.out_of_context_harness("core", ports)
    assert (input_bits, output_bits) == (4, 2)
    assert (
        ".a(in_sr[2:0])" in harness
        and ".b(in_sr[3:3])" in harness
        and ".y(core_out[1:0])" in harness
    )
    assert ".clk(clk)" in harness and ".rst_n(rst_n)" in harness
    assert (
        "module sc_ooc_harness(input wire clk, input wire rst_n, input wire sin, input wire load"
        in harness
    )
    assert stub.startswith("module core(") and "output wire [1:0] y" in stub


def test_only_transitively_instanced_primitives_are_appended() -> None:
    """A dense layer pulls its primitives in; an unused primitive stays out."""
    verilog, _ = _dense(2, 2).emit_sv_with_report("ice40:UP5K", 12)
    source = feedback.with_instanced_primitives(verilog)
    for module in (
        "sc_dense_matrix_layer",
        "sc_bitstream_encoder",
        "sc_lif_neuron",
        "sc_dotproduct_to_current",
    ):
        assert f"module {module}" in source
    assert "module sc_firing_rate_bank" not in source


def test_unsupported_target_and_clock_are_refused() -> None:
    """Only the calibrated device and a positive clock are accepted."""
    with pytest.raises(ValueError, match="resource feedback supports"):
        feedback.run_resource_feedback(
            _dense(2, 2), clock_mhz=12, target="ecp5", timeout_seconds=TOOL_TIMEOUT_S
        )
    with pytest.raises(ValueError, match="clock_mhz must be positive"):
        feedback.run_resource_feedback(_dense(2, 2), clock_mhz=0, timeout_seconds=TOOL_TIMEOUT_S)


@pytest.mark.parametrize(
    ("returncode", "log", "error"),
    [
        (
            1,
            "ERROR: Unable to find a placement location for cell 'y$sb_io'\n",
            "the design needs more I/O pins than the package has; "
            "wrap the core in an I/O harness or choose a larger package",
        ),
        (
            1,
            "ERROR: Unable to place cell 'x_SB_LUT4', no BELs remaining to implement cell"
            " type 'ICESTORM_LC'\n",
            "the design needs more logic than the device has",
        ),
        (
            1,
            "ERROR: Failed to expand region (0, 0) |_> (25, 31) of 5404 ICESTORM_LCs\n",
            "the design needs more logic than the device has",
        ),
        (1, "ERROR: routing failed\n", None),
        # Yosys 0.70 debug cells left in a netlist: not a capacity failure.
        (
            1,
            "ERROR: Unable to place cell 'u_dense_0.NEURON[1].u_lif', no BELs remaining to"
            " implement cell type '$scopeinfo'\n",
            None,
        ),
        (0, "Info: placed 'y$sb_io'\nMax frequency: 30.0 MHz\n", None),
    ],
)
def test_studio_pnr_names_a_package_pin_shortage_and_a_capacity_failure(
    returncode: int, log: str, error: str | None, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unplaceable I/O buffer is too few pins; a capacity marker is too little logic."""
    netlist = tmp_path / "design.json"
    netlist.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(synthesis, "_resolve_eda_tool", lambda name: "/bin/true")
    monkeypatch.setattr(
        synthesis,
        "_run_eda_command",
        lambda args, **kwargs: subprocess.CompletedProcess(args, returncode, stdout=log, stderr=""),
    )
    result = synthesis.run_pnr(str(netlist), "ice40")
    assert result["success"] is (returncode == 0)
    assert result.get("error") == error


def test_missing_tools_are_undetermined(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Without Yosys and nextpnr on PATH the run is undetermined, never a pass."""
    monkeypatch.setenv("PATH", str(tmp_path))
    result = feedback.run_resource_feedback(
        _dense(2, 2), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    assert result["verdict"] == "undetermined"
    assert _verdicts(result) == {"toolchain": "undetermined"}


@real_flow
def test_a_small_dense_layer_passes_every_check_within_its_estimate() -> None:
    """2x2 dense: ports fit SG48, logic and 12 MHz timing hold, within the bound."""
    result = feedback.run_resource_feedback(
        _dense(2, 2), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    assert _verdicts(result) == {
        "io_fits_package": "pass",
        "logic_fits_device": "pass",
        "timing": "pass",
        "estimate": "pass",
    }
    assert result["verdict"] == "pass"
    assert "$scopeinfo" not in result["synthesis"]["cells"]
    assert 0 < result["core_logic_cells"] <= result["estimate"]["logic_cells_estimated"]


@real_flow
def test_a_layer_above_the_device_capacity_is_a_logic_violation() -> None:
    """12x8 dense needs more logic cells than the UP5K has."""
    result = feedback.run_resource_feedback(
        _dense(12, 8), clock_mhz=12, timeout_seconds=2 * TOOL_TIMEOUT_S
    )
    assert _verdicts(result)["logic_fits_device"] == "violation"
    assert result["verdict"] == "violation"


@real_flow
def test_a_clock_above_the_measured_fmax_is_a_timing_violation() -> None:
    """4x3 dense routes near 26 MHz, so 100 MHz misses timing but not the estimate."""
    result = feedback.run_resource_feedback(
        _dense(4, 3), clock_mhz=100, timeout_seconds=TOOL_TIMEOUT_S
    )
    verdicts = _verdicts(result)
    assert verdicts["timing"] == "violation"
    assert verdicts["estimate"] == "pass"
    assert result["verdict"] == "violation"


@real_flow
def test_a_real_run_above_a_lowered_estimate_is_an_estimate_violation() -> None:
    """The comparison fails when the routed core exceeds the estimate it is given."""
    result = feedback.run_resource_feedback(
        _LoweredEstimate(_dense(2, 2), 10), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    verdicts = _verdicts(result)
    assert verdicts["estimate"] == "violation"
    assert verdicts["logic_fits_device"] == "pass"


@real_flow
def test_a_word_graph_is_judged_by_its_routed_result_alone() -> None:
    """No static estimate: no estimate check, and the report says why."""
    builder = ScGraphBuilder("word_add")
    a = builder.input("a", "i16")
    builder.output("doubled", builder.word_binary("add", a, a, "i16"))
    result = feedback.run_resource_feedback(
        builder.build(), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    assert "estimate" not in _verdicts(result)
    assert result["estimate_comparison"].startswith("not applicable: word-level operations")
    assert result["verdict"] == "pass"


@real_flow
def test_a_toolchain_other_than_the_calibration_makes_the_comparison_undetermined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real run under a different named toolchain cannot pass the estimate check."""
    monkeypatch.setitem(feedback.CALIBRATION_TOOLCHAIN, "yosys", "Yosys 9.99")
    result = feedback.run_resource_feedback(
        _dense(2, 2), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    assert _verdicts(result)["estimate"] == "undetermined"
    assert result["verdict"] == "undetermined"


@real_flow
def test_the_core_logic_cell_rule_reproduces_a_recorded_design() -> None:
    """Routed cells minus harness flip-flops equal the recorded core of the same design."""
    record = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    design = next(d for d in record["designs"] if d["name"] == "fit_dense_2x2_w12")
    result = feedback.run_resource_feedback(
        _graph(design), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    assert result["core_logic_cells"] == design["core_logic_cells"]
    harness = result["harness"]
    assert harness["logic_cells"] == harness["input_bits"] + harness["output_bits"]


@real_flow
def test_weights_that_quantise_to_zero_only_remove_logic() -> None:
    """Below 1/256 a rate weight is code 0 and costs nothing; every non-zero code costs the same."""

    def core(weight: float) -> int:
        graph = _dense(2, 2, 12, [weight] * 4)
        result = feedback.run_resource_feedback(graph, clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S)
        cells = result["core_logic_cells"]
        assert isinstance(cells, int)
        return cells

    zero, below_one_code, one_code, full = core(0.0), core(0.99 / 256), core(1 / 256), core(1.0)
    one_weight = feedback.run_resource_feedback(
        _dense(2, 2, 12, [0.7, 0.0, 0.0, 0.0]), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )["core_logic_cells"]
    assert zero == below_one_code < one_weight < one_code == full


@real_flow
def test_a_fitting_core_whose_harness_overflows_is_undetermined_with_its_size() -> None:
    """12x4 w12 packs 5404 cells with its harness; its 5208-cell core is reported, not judged.

    Its ports exceed the SG48 pins, so the overall verdict is that I/O violation.
    """
    record = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    design = next(d for d in record["designs"] if d["name"] == "fit_dense_12x4_w12")
    result = feedback.run_resource_feedback(
        _graph(design), clock_mhz=12, timeout_seconds=2 * TOOL_TIMEOUT_S
    )
    assert result["core_logic_cells"] == design["packed_core_logic_cells"] == 5208
    assert _verdicts(result) == {
        "io_fits_package": "violation",
        "logic_fits_device": "undetermined",
    }
    assert result["verdict"] == "violation"


def test_tool_timeouts_and_errors_are_recorded_not_raised(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Yosys timeout or error and a nextpnr timeout each become a recorded failure."""

    def timeout(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args, 5)

    monkeypatch.setattr(feedback, "_run_eda_command", timeout)
    with pytest.raises(feedback._ToolFailure, match="yosys exceeded 5 s"):
        feedback._yosys("yosys", "synth", tmp_path, 5, None)
    assert feedback._place_and_route("nextpnr-ice40", tmp_path, 12, 5, None) == {
        "success": False,
        "log_excerpt": "nextpnr exceeded 5 s",
    }
    failed = subprocess.CompletedProcess(["yosys"], 1, stdout="ok\nERROR: no top\n", stderr="")
    monkeypatch.setattr(feedback, "_run_eda_command", lambda args, **kwargs: failed)
    with pytest.raises(feedback._ToolFailure, match="^ERROR: no top$"):
        feedback._yosys("yosys", "synth", tmp_path, 5, None)


@real_flow
@pytest.mark.parametrize("failing_call", [1, 2])
def test_a_failed_synthesis_or_harness_step_is_undetermined(
    failing_call: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first real Yosys run that fails stops the check as undetermined at that step."""
    real = feedback._yosys
    calls: list[int] = []

    def flaky(*args: Any) -> None:
        calls.append(1)
        if len(calls) == failing_call:
            raise feedback._ToolFailure("planted Yosys failure")
        real(*args)

    monkeypatch.setattr(feedback, "_yosys", flaky)
    result = feedback.run_resource_feedback(
        _dense(2, 2), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    step = "synthesis" if failing_call == 1 else "harness"
    assert result["checks"][-1] == {
        "name": step,
        "verdict": "undetermined",
        "detail": "planted Yosys failure",
    }
    assert result["verdict"] == "undetermined"


@real_flow
def test_a_place_and_route_failure_without_a_capacity_marker_is_undetermined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A routing failure says nothing about capacity, so capacity stays undetermined."""
    monkeypatch.setattr(
        feedback,
        "_place_and_route",
        lambda *args: {
            "success": False,
            "log_excerpt": "ERROR: routing failed",
            "packed_logic_cells": 900,
        },
    )
    result = feedback.run_resource_feedback(
        _dense(2, 2), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    assert result["checks"][-1] == {
        "name": "logic_fits_device",
        "verdict": "undetermined",
        "detail": "ERROR: routing failed",
    }
    assert "core_logic_cells" not in result


@real_flow
def test_studio_ice40_synthesis_leaves_no_scopeinfo_for_nextpnr(tmp_path: Any) -> None:
    """Yosys 0.70's $scopeinfo cells are deleted, so nextpnr 0.6 places the design.

    Left in, nextpnr stops with "no BELs remaining to implement cell type
    '$scopeinfo'" on this same 2x2 layer.
    """
    source = feedback.with_instanced_primitives(_dense(2, 2).emit_sv())
    result, json_path = synthesis._run_synthesis_in_directory(
        source, "ice40", root=tmp_path, process_limits=None, target_provenance={}
    )
    assert result["success"] is True and json_path is not None
    netlist = json.loads(json_path.read_text(encoding="utf-8"))
    types = {c["type"] for m in netlist["modules"].values() for c in m.get("cells", {}).values()}
    assert "$scopeinfo" not in types
    routed = synthesis.run_pnr(str(json_path), "ice40")
    assert routed["success"] is True, routed.get("log_excerpt")


@real_flow
def test_a_layer_with_port_weights_is_judged_by_its_routed_result_alone() -> None:
    """Port weights build a runtime multiplier; there is no bound, only the routed checks."""
    result = feedback.run_resource_feedback(
        _dense_with_ports(True, False), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    assert "estimate" not in _verdicts(result)
    assert result["estimate_comparison"].startswith(
        "not applicable: operand source, type or parameter outside the calibration"
    )
    assert result["core_logic_cells"] > 901


@pytest.mark.parametrize(
    ("behaviour", "detail"),
    [
        ("timeout", "/bin/yosys -V exceeded 5.0 s"),
        ("empty", "/bin/yosys -V printed no version"),
        ("missing", "/bin/yosys -V cannot run: no such file"),
    ],
)
def test_a_tool_version_that_cannot_be_read_is_undetermined(
    behaviour: str, detail: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung, silent or vanished tool leaves the toolchain undetermined, never raises."""

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if behaviour == "timeout":
            raise subprocess.TimeoutExpired(args, 5)
        if behaviour == "missing":
            raise FileNotFoundError("no such file")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(feedback, "_resolve_eda_tool", lambda name: f"/bin/{name}")
    monkeypatch.setattr(feedback, "_run_eda_command", run)
    result = feedback.run_resource_feedback(_dense(2, 2), clock_mhz=12, timeout_seconds=5.0)
    assert result["checks"] == [{"name": "toolchain", "verdict": "undetermined", "detail": detail}]
    assert result["verdict"] == "undetermined"
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        feedback.run_resource_feedback(_dense(2, 2), clock_mhz=12, timeout_seconds=0)


@real_flow
def test_a_routed_design_without_a_logic_cell_count_is_undetermined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing that reports no ICESTORM_LC utilisation cannot pass the capacity check."""
    monkeypatch.setattr(
        feedback,
        "_place_and_route",
        lambda *args: {
            "success": True,
            "utilisation": {},
            "fmax": {"clk": {"achieved": 30.0, "constraint": 12.0}},
            "timing_met": True,
        },
    )
    result = feedback.run_resource_feedback(
        _dense(2, 2), clock_mhz=12, timeout_seconds=TOOL_TIMEOUT_S
    )
    verdicts = _verdicts(result)
    assert verdicts["logic_fits_device"] == "undetermined"
    assert verdicts["estimate"] == "undetermined"
    assert result["core_logic_cells"] is None
