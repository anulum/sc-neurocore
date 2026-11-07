# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resource and timing feedback of a compiled graph on a real FPGA flow

"""Synthesise, place and route one IR graph and judge it against its device.

The emitted top level of a graph exposes more ports than a small FPGA package
has pins, so placement and routing run out of context: a harness feeds every
input from a serial shift register and captures every output in a
parallel-load shift register, five pins in all. The harness is synthesised
against a port stub of the core and merged with the core's own netlist, so the
core is mapped exactly once. The core's logic cells are the routed total less
the harness, which takes one flip-flop per input bit and one flip-flop plus one
multiplexer per output bit.
"""

from __future__ import annotations

import json
import re
import subprocess  # nosec B404
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from sc_neurocore.hdl.resources import baseline_primitive_text, list_baseline_primitive_rtl
from sc_neurocore.studio.synthesis import (
    EdaProcessLimits,
    _resolve_eda_tool,
    capacity_failure,
    _run_eda_command,
)

#: Engine device target for each Studio target this feedback supports.
ENGINE_TARGETS: dict[str, str] = {"ice40": "ice40:UP5K"}

#: Logic cells of the iCE40 UltraPlus UP5K (Lattice FPGA-DS-02008).
ICE40_UP5K_LOGIC_CELLS = 5280

#: User I/O pins of the UP5K SG48 package that nextpnr-ice40 0.6-3build5
#: places: one clock input and 38 outputs, measured 2026-11-07.
ICE40_UP5K_SG48_IO_PINS = 39

#: Tool versions the engine's iCE40 calibration was measured with. Another
#: version makes the estimate comparison undetermined, never a pass.
CALIBRATION_TOOLCHAIN = {
    "yosys": "Yosys 0.70+post (git sha1 0c11ea098be065e585a90a06e7b7da49936cf231",
    "nextpnr-ice40": "Version 0.6-3build5",
}

SCHEMA_VERSION = "sc-neurocore.resource-feedback.v1"

PASS, VIOLATION, UNDETERMINED = "pass", "violation", "undetermined"

#: nextpnr's device utilisation line for packed logic cells.
_PACKED_LOGIC_CELLS = re.compile(r"ICESTORM_LC:\s+(\d+)/\s*\d+")


class FeedbackGraph(Protocol):
    """The graph surface the feedback needs from ``sc_neurocore_engine.ir.ScGraph``."""

    @property
    def name(self) -> str:
        """Module name of the emitted top level."""
        ...

    def emit_sv_with_report(self, target: str, clock_mhz: int) -> tuple[str, Mapping[str, Any]]:
        """Emit SystemVerilog for a device with the engine's static estimate."""
        ...


def with_instanced_primitives(verilog: str) -> str:
    """Append every baseline primitive the design instances, transitively.

    Only instanced primitives are read: an unused primitive with an
    unpacked-array port would stop Yosys's Verilog reader.

    Parameters
    ----------
    verilog : str
        Emitted top level.

    Returns
    -------
    str
        The top level followed by the primitives it needs.
    """
    texts = {
        name.removesuffix(".v"): baseline_primitive_text(name)
        for name in list_baseline_primitive_rtl()
    }
    used: list[str] = []
    source = verilog
    changed = True
    while changed:
        changed = False
        for stem, text in texts.items():
            if stem not in used and re.search(rf"\b{re.escape(stem)}\b\s*(#|\w)", source):
                used.append(stem)
                source += text
                changed = True
    return source


def out_of_context_harness(
    top: str, ports: Mapping[str, Mapping[str, Any]]
) -> tuple[str, str, int, int]:
    """Write a five-pin harness and a port stub for one core.

    Parameters
    ----------
    top : str
        Core module name.
    ports : Mapping
        The core's ports from its Yosys JSON netlist (``direction`` and ``bits``).

    Returns
    -------
    tuple of (str, str, int, int)
        Harness Verilog, core port stub, input bits and output bits it shifts.
    """
    inputs = [
        (n, len(p["bits"]))
        for n, p in ports.items()
        if p["direction"] == "input" and n not in ("clk", "rst_n")
    ]
    outputs = [(n, len(p["bits"])) for n, p in ports.items() if p["direction"] == "output"]
    input_bits = max(1, sum(w for _, w in inputs))
    output_bits = max(1, sum(w for _, w in outputs))
    connections = [f".{n}({n})" for n in ("clk", "rst_n") if n in ports]
    low = 0
    for name, w in inputs:
        connections.append(f".{name}(in_sr[{low + w - 1}:{low}])")
        low += w
    low = 0
    for name, w in outputs:
        connections.append(f".{name}(core_out[{low + w - 1}:{low}])")
        low += w
    harness = "\n".join(
        [
            "module sc_ooc_harness(input wire clk, input wire rst_n, input wire sin, input wire load, output wire sout);",
            f"    reg [{input_bits - 1}:0] in_sr;",
            "    always @(posedge clk) in_sr <= {in_sr, sin};",
            f"    wire [{output_bits - 1}:0] core_out;",
            f"    reg [{output_bits - 1}:0] out_sr;",
            "    always @(posedge clk) out_sr <= load ? core_out : (out_sr << 1);",
            f"    assign sout = out_sr[{output_bits - 1}];",
            f"    {top} u_core({', '.join(connections)});",
            "endmodule",
            "",
        ]
    )
    declarations = [
        f"    {p['direction']} wire [{len(p['bits']) - 1}:0] {n}" for n, p in ports.items()
    ]
    stub = f"module {top}(\n" + ",\n".join(declarations) + "\n);\nendmodule\n"
    return harness, stub, input_bits, output_bits


class _ToolFailure(Exception):
    """A tool run that ended without a usable result."""


def _yosys(
    yosys: str, script: str, root: Path, timeout: float, limits: EdaProcessLimits | None
) -> None:
    """Run one Yosys script in ``root``.

    Raises
    ------
    _ToolFailure
        With Yosys's error lines, or on timeout.
    """
    try:
        completed = _run_eda_command(
            [yosys, "-q", "-p", script], cwd=root, timeout_seconds=timeout, process_limits=limits
        )
    except subprocess.TimeoutExpired:
        raise _ToolFailure(f"yosys exceeded {timeout} s") from None
    if completed.returncode != 0:
        log = completed.stdout + completed.stderr
        errors = [line.strip() for line in log.splitlines() if "ERROR" in line]
        raise _ToolFailure("; ".join(errors[:3]) or log[-500:])


def _version(executable: str, timeout: float) -> str:
    """Return the first line a tool prints for ``--version``/``-V``.

    Raises
    ------
    _ToolFailure
        When the tool cannot run, exceeds ``timeout`` or prints nothing.
    """
    flag = "-V" if Path(executable).name == "yosys" else "--version"
    try:
        completed = _run_eda_command([executable, flag], timeout_seconds=timeout)
    except subprocess.TimeoutExpired:
        raise _ToolFailure(f"{executable} {flag} exceeded {timeout} s") from None
    except OSError as error:
        raise _ToolFailure(f"{executable} {flag} cannot run: {error}") from None
    lines = (completed.stdout + completed.stderr).strip().splitlines()
    if not lines:
        raise _ToolFailure(f"{executable} {flag} printed no version")
    return lines[0]


def _place_and_route(
    nextpnr: str, root: Path, clock_mhz: int, timeout: float, limits: EdaProcessLimits | None
) -> dict[str, Any]:
    """Place and route ``harness.json`` on the UP5K SG48 for a clock.

    A missed constraint does not stop routing, so a routed design always has
    its Fmax. Returns ``success``, the routed ``utilisation`` per cell type,
    ``fmax`` per clock, ``timing_met`` and the end of the log.
    """
    try:
        completed = _run_eda_command(
            [
                nextpnr,
                "--up5k",
                "--package",
                "sg48",
                "--json",
                "harness.json",
                "--freq",
                str(clock_mhz),
                "--timing-allow-fail",
                "--report",
                "report.json",
                "--asc",
                "harness.asc",
            ],
            cwd=root,
            timeout_seconds=timeout,
            process_limits=limits,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "log_excerpt": f"nextpnr exceeded {timeout} s"}
    log = completed.stdout + completed.stderr
    outcome: dict[str, Any] = {"success": completed.returncode == 0, "log_excerpt": log[-600:]}
    # nextpnr packs before it places and reports the packed logic cells either way.
    packed = _PACKED_LOGIC_CELLS.search(log)
    outcome["packed_logic_cells"] = None if packed is None else int(packed.group(1))
    report_path = root / "report.json"
    if completed.returncode == 0 and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # nextpnr names the field in American spelling.
        outcome["utilisation"] = report.get("utilization")
        clocks = report.get("fmax") or {}
        outcome["fmax"] = clocks
        outcome["timing_met"] = timing_met(clocks)
    return outcome


def timing_met(clocks: Mapping[str, Mapping[str, float]]) -> bool | None:
    """Return whether every clock reaches its constraint, or None without clocks.

    An achieved frequency equal to the constraint meets it.
    """
    if not clocks:
        return None
    return all(row["achieved"] >= row["constraint"] for row in clocks.values())


def capacity_check(core: int, packed: int) -> dict[str, Any]:
    """Judge a design whose core and harness did not fit the device.

    The core needs ``packed`` minus the harness flip-flops. A core above the
    device's logic cells is a violation. A core that fits while the core and
    its harness do not is undetermined: nothing was routed, so placement and
    timing of the core alone are unknown.
    """
    if core > ICE40_UP5K_LOGIC_CELLS:
        return _check(
            "logic_fits_device",
            VIOLATION,
            f"the core needs {core} of {ICE40_UP5K_LOGIC_CELLS} logic cells",
        )
    return _check(
        "logic_fits_device",
        UNDETERMINED,
        f"the core needs {core} of {ICE40_UP5K_LOGIC_CELLS} logic cells, but with its "
        f"five-pin harness ({packed}) it does not fit, so nothing was routed",
    )


def io_check(port_bits: int) -> dict[str, Any]:
    """Judge the core's port bits against the SG48 user pins; equal fits."""
    return _check(
        "io_fits_package",
        PASS if port_bits <= ICE40_UP5K_SG48_IO_PINS else VIOLATION,
        f"{port_bits} port bits for {ICE40_UP5K_SG48_IO_PINS} SG48 pins",
    )


def _cells(netlist: Mapping[str, Any]) -> dict[str, int]:
    """Count the cells of every non-library module in a Yosys JSON netlist."""
    counts: dict[str, int] = {}
    for module in netlist["modules"].values():
        attributes = module.get("attributes", {})
        if "blackbox" in attributes or "whitebox" in attributes:
            continue
        for cell in module.get("cells", {}).values():
            counts[cell["type"]] = counts.get(cell["type"], 0) + 1
    return counts


def _check(name: str, verdict: str, detail: object) -> dict[str, Any]:
    """Build one named check."""
    return {"name": name, "verdict": verdict, "detail": detail}


def overall_verdict(checks: Sequence[Mapping[str, Any]]) -> str:
    """Combine checks: any violation wins, then any undetermined, else pass.

    Parameters
    ----------
    checks : Sequence of Mapping
        Checks with a ``verdict`` of pass, violation or undetermined.

    Returns
    -------
    str
        The combined verdict; an empty list is undetermined.
    """
    verdicts = [check["verdict"] for check in checks]
    if VIOLATION in verdicts:
        return VIOLATION
    if UNDETERMINED in verdicts or not verdicts:
        return UNDETERMINED
    return PASS


def run_resource_feedback(
    graph: FeedbackGraph,
    *,
    clock_mhz: int,
    target: str = "ice40",
    timeout_seconds: float,
    process_limits: EdaProcessLimits | None = None,
) -> dict[str, Any]:
    """Synthesise, place and route a graph out of context and judge it.

    Each check is a pass, a violation or undetermined. A missing tool, a
    timeout or a tool error is undetermined and never a pass.

    Parameters
    ----------
    graph : FeedbackGraph
        A verified IR graph.
    clock_mhz : int
        Target clock in MHz; must be positive.
    target : str, default "ice40"
        Studio target; one of :data:`ENGINE_TARGETS`.
    timeout_seconds : float
        Wall-clock limit for each Yosys and nextpnr run, positive. The caller
        sets it for the host that runs the tools; no host-independent default
        exists.
    process_limits : EdaProcessLimits or None
        Optional CPU and memory ceilings for the tool processes.

    Returns
    -------
    dict
        ``schema_version``, ``target``, ``device``, ``clock_mhz``, the engine
        ``estimate``, ``toolchain`` versions, ``synthesis`` cell counts of the
        core, ``io`` (port bits against package pins), ``harness``,
        ``place_and_route``, ``core_logic_cells``, ``checks`` and ``verdict``.

    Raises
    ------
    ValueError
        For an unsupported target, a non-positive clock or a non-positive
        timeout.
    """
    if target not in ENGINE_TARGETS:
        raise ValueError(f"resource feedback supports {sorted(ENGINE_TARGETS)}, not {target!r}")
    if clock_mhz <= 0:
        raise ValueError("clock_mhz must be positive")
    if not timeout_seconds > 0:
        raise ValueError("timeout_seconds must be positive")
    verilog, estimate = graph.emit_sv_with_report(ENGINE_TARGETS[target], clock_mhz)
    checks: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "device": estimate["device_part"],
        "clock_mhz": clock_mhz,
        "estimate": dict(estimate),
        "checks": checks,
    }
    yosys = _resolve_eda_tool("yosys")
    nextpnr = _resolve_eda_tool("nextpnr-ice40")
    if yosys is None or nextpnr is None:
        missing = [
            name for name, path in (("yosys", yosys), ("nextpnr-ice40", nextpnr)) if path is None
        ]
        checks.append(_check("toolchain", UNDETERMINED, f"not installed: {', '.join(missing)}"))
        result["verdict"] = overall_verdict(checks)
        return result
    try:
        toolchain = {
            "yosys": _version(yosys, timeout_seconds),
            "nextpnr-ice40": _version(nextpnr, timeout_seconds),
        }
    except _ToolFailure as failure:
        checks.append(_check("toolchain", UNDETERMINED, str(failure)))
        result["verdict"] = overall_verdict(checks)
        return result
    result["toolchain"] = toolchain
    calibrated = all(CALIBRATION_TOOLCHAIN[name] in version for name, version in toolchain.items())
    top = graph.name
    with tempfile.TemporaryDirectory(prefix="sc_resource_feedback_") as tmpdir:
        root = Path(tmpdir)
        (root / "design.v").write_text(with_instanced_primitives(verilog), encoding="utf-8")
        try:
            _yosys(
                yosys,
                f"read_verilog -sv design.v; synth_ice40 -top {top}; delete t:$scopeinfo; write_json design.json",
                root,
                timeout_seconds,
                process_limits,
            )
        except _ToolFailure as failure:
            checks.append(_check("synthesis", UNDETERMINED, str(failure)))
            result["verdict"] = overall_verdict(checks)
            return result
        netlist = json.loads((root / "design.json").read_text(encoding="utf-8"))
        cells = _cells(netlist)
        result["synthesis"] = {"cells": cells}
        ports = netlist["modules"][top]["ports"]
        port_bits = sum(len(port["bits"]) for port in ports.values())
        result["io"] = {
            "port_bits": port_bits,
            "package": "SG48",
            "package_pins": ICE40_UP5K_SG48_IO_PINS,
        }
        checks.append(io_check(port_bits))
        harness, stub, input_bits, output_bits = out_of_context_harness(top, ports)
        (root / "harness.v").write_text(harness, encoding="utf-8")
        (root / "core_stub.v").write_text(stub, encoding="utf-8")
        try:
            _yosys(
                yosys,
                "read_verilog -lib core_stub.v; read_verilog -sv harness.v; synth_ice40 -top sc_ooc_harness; "
                "delete t:$scopeinfo; select sc_ooc_harness; write_json -selected harness_only.json",
                root,
                timeout_seconds,
                process_limits,
            )
            _yosys(
                yosys,
                "read_json design.json; read_json harness_only.json; hierarchy -top sc_ooc_harness; flatten; "
                "delete t:$scopeinfo; write_json harness.json",
                root,
                timeout_seconds,
                process_limits,
            )
        except _ToolFailure as failure:
            checks.append(_check("harness", UNDETERMINED, str(failure)))
            result["verdict"] = overall_verdict(checks)
            return result
        harness_netlist = json.loads((root / "harness_only.json").read_text(encoding="utf-8"))
        harness_cells = _cells(harness_netlist)
        harness_flip_flops = sum(
            n for kind, n in harness_cells.items() if kind.startswith("SB_DFF")
        )
        pnr = _place_and_route(nextpnr, root, clock_mhz, timeout_seconds, process_limits)
    result["harness"] = {
        "input_bits": input_bits,
        "output_bits": output_bits,
        "cells": harness_cells,
        "logic_cells": harness_flip_flops,
        "method": "each harness multiplexer feeds its own flip-flop and packs into that logic cell",
    }
    result["place_and_route"] = {
        key: pnr.get(key) for key in ("success", "utilisation", "fmax", "timing_met")
    }
    if not pnr.get("success"):
        log = str(pnr.get("log_excerpt") or "")
        packed = pnr.get("packed_logic_cells")
        if capacity_failure(log) and isinstance(packed, int):
            checks.append(capacity_check(packed - harness_flip_flops, packed))
            result["core_logic_cells"] = packed - harness_flip_flops
        else:
            checks.append(_check("logic_fits_device", UNDETERMINED, log[-300:]))
        result["verdict"] = overall_verdict(checks)
        return result
    used = ((pnr.get("utilisation") or {}).get("ICESTORM_LC") or {}).get("used")
    core = used - harness_flip_flops if isinstance(used, int) else None
    result["core_logic_cells"] = core
    if isinstance(used, int):
        checks.append(
            _check(
                "logic_fits_device",
                PASS,
                f"{used} of {ICE40_UP5K_LOGIC_CELLS} logic cells with harness",
            )
        )
    else:
        checks.append(
            _check(
                "logic_fits_device",
                UNDETERMINED,
                "routed, but nextpnr reported no ICESTORM_LC utilisation",
            )
        )
    timing = pnr.get("timing_met")
    checks.append(
        _check(
            "timing",
            PASS if timing is True else VIOLATION if timing is False else UNDETERMINED,
            pnr.get("fmax"),
        )
    )
    if estimate.get("static_estimate"):
        checks.append(_estimate_check(estimate, core, pnr.get("fmax"), calibrated))
        result["estimate_comparison"] = "compared with the calibrated static estimate"
    else:
        result["estimate_comparison"] = (
            f"not applicable: {estimate.get('static_estimate_reason')}; the routed result alone decides"
        )
    result["verdict"] = overall_verdict(checks)
    return result


def _estimate_check(
    estimate: Mapping[str, Any],
    core: int | None,
    fmax: Mapping[str, Mapping[str, float]] | None,
    calibrated: bool,
) -> dict[str, Any]:
    """Judge the engine's static estimate against the routed core.

    The caller compares only a report whose ``static_estimate`` is true.
    """
    if not calibrated:
        return _check("estimate", UNDETERMINED, "calibration toolchain mismatch")
    if core is None or not fmax:
        return _check("estimate", UNDETERMINED, "routed core size or Fmax missing")
    worst_path_ns = max(1000.0 / row["achieved"] for row in fmax.values())
    logic_bound = estimate["logic_cells_estimated"]
    path_bound = estimate["critical_path_estimate_ns"]
    exceeded = []
    if core > logic_bound:
        exceeded.append(f"core logic cells {core} > estimate {logic_bound}")
    if worst_path_ns > path_bound:
        exceeded.append(f"critical path {worst_path_ns:.2f} ns > estimate {path_bound:.2f} ns")
    return _check(
        "estimate",
        VIOLATION if exceeded else PASS,
        "; ".join(exceeded) or "within the calibrated bound",
    )
