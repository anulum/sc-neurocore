# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synthesis Dashboard backend for Studio

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from math import ceil
from typing import Any
import json
import os
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path

from sc_neurocore.studio.synthesis_provenance import (
    ToolStatusMap,
    build_synthesis_target_provenance,
    build_synthesis_target_provenance_matrix,
)

_EDA_TOOL_ALLOWLIST = frozenset({"yosys", "nextpnr-ice40", "nextpnr-ecp5", "firtool"})
_MAX_TERMINAL_ARTIFACT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EdaProcessLimits:
    """Optional process resource limits for external Studio EDA commands.

    Parameters
    ----------
    cpu_seconds:
        Maximum CPU seconds allowed for the child process on hosts that expose
        POSIX ``RLIMIT_CPU``. ``None`` leaves CPU accounting to the existing
        wall-clock timeout.
    address_space_bytes:
        Maximum address space bytes allowed for the child process on hosts that
        expose POSIX ``RLIMIT_AS``. ``None`` leaves memory unconstrained by this
        helper.
    """

    cpu_seconds: float | None = None
    address_space_bytes: int | None = None

    def __post_init__(self) -> None:
        """Validate positive resource ceilings when they are configured."""

        if self.cpu_seconds is not None and self.cpu_seconds <= 0:
            raise ValueError("EDA process CPU limit must be positive.")
        if self.address_space_bytes is not None and self.address_space_bytes <= 0:
            raise ValueError("EDA process memory limit must be positive.")


@dataclass(frozen=True, slots=True)
class SynthesisTerminalExecution:
    """Public terminal report plus private implementation artifacts."""

    netlist_json: bytes | None
    report: dict[str, Any]
    routed_design: bytes | None


def _resolve_eda_tool(name: str) -> str | None:
    """Resolve an allowlisted EDA executable to an absolute path."""

    if name not in _EDA_TOOL_ALLOWLIST:
        raise ValueError(f"Unsupported EDA tool: {name}")
    return shutil.which(name)


def _eda_process_limits_supported() -> bool:
    """Return whether this host can apply POSIX child-process limits."""

    return os.name == "posix"


def _build_limit_preexec(limits: EdaProcessLimits | None) -> Callable[[], None] | None:
    """Build a POSIX pre-exec hook that applies configured EDA limits."""

    if limits is None or not _eda_process_limits_supported():
        return None
    if limits.cpu_seconds is None and limits.address_space_bytes is None:
        return None

    def apply_limits() -> None:
        import resource

        if limits.cpu_seconds is not None:
            cpu_limit = max(1, ceil(limits.cpu_seconds))
            _, hard = resource.getrlimit(resource.RLIMIT_CPU)
            if hard != resource.RLIM_INFINITY:
                cpu_limit = min(cpu_limit, hard)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        if limits.address_space_bytes is not None and hasattr(resource, "RLIMIT_AS"):
            memory_limit = int(limits.address_space_bytes)
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            if hard != resource.RLIM_INFINITY:
                memory_limit = min(memory_limit, hard)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

    return apply_limits


def _run_eda_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: float,
    process_limits: EdaProcessLimits | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one allowlisted EDA command with optional child-process limits."""

    limit_preexec = _build_limit_preexec(process_limits)
    if limit_preexec is None:
        return subprocess.run(  # nosec B603
            list(command),
            capture_output=True,
            cwd=cwd,
            shell=False,
            text=True,
            timeout=timeout_seconds,
        )
    return subprocess.run(  # nosec B603
        list(command),
        capture_output=True,
        cwd=cwd,
        preexec_fn=limit_preexec,
        shell=False,
        text=True,
        timeout=timeout_seconds,
    )


def check_tools() -> dict[str, Any]:
    """Detect which EDA tools are installed."""
    tools: dict[str, dict[str, bool | str | None]] = {}
    for name, cmd in [
        ("yosys", ["yosys", "--version"]),
        ("nextpnr_ice40", ["nextpnr-ice40", "--version"]),
        ("nextpnr_ecp5", ["nextpnr-ecp5", "--version"]),
        ("firtool", ["firtool", "--version"]),
    ]:
        try:
            executable = _resolve_eda_tool(cmd[0])
            if executable is None:
                tools[name] = {"available": False, "version": None}
                continue
            r = _run_eda_command([executable, *cmd[1:]], timeout_seconds=5)
            version_lines = (r.stdout + "\n" + r.stderr).strip().splitlines()
            version = version_lines[0] if r.returncode == 0 and version_lines else None
            tools[name] = {"available": r.returncode == 0, "version": version}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            tools[name] = {"available": False, "version": None}
    return tools


_TARGETS: dict[str, dict[str, str | None]] = {
    "ice40": {"synth_cmd": "synth_ice40", "pnr": "nextpnr-ice40", "device": "up5k"},
    "ecp5": {
        "synth_cmd": "synth_ecp5",
        "pnr": "nextpnr-ecp5",
        "device": "25k",
        "package": "CABGA381",
    },
    "gowin": {"synth_cmd": "synth_gowin", "pnr": None, "device": None},
    "xilinx": {"synth_cmd": "synth_xilinx -flatten", "pnr": None, "device": None},
}

#: Resource capacity of the device each target is judged against, and that
#: device's name. Gowin and Xilinx synthesis here is not device-bound, so the
#: device is named with the verdict rather than implied by the family.
#:
#: - iCE40 UltraPlus UP5K: 5280 LUT4 and flip-flops, 30 EBR, 8 SB_MAC16
#:   (Lattice FPGA-DS-02008; nextpnr-ice40 --up5k reports the same).
#: - ECP5 LFE5U-25F: 24288 LUT4 and flip-flops, 56 DP16KD, 28 MULT18X18D, as
#:   nextpnr-ecp5 --25k reports them (the data sheet rounds to "24K"). The row
#:   said 24576.
#: - Gowin GW2A-18: 20736 LUT4, 15552 flip-flops, 46 B-SRAM, 48 18x18
#:   multipliers (Gowin DS102). The row said 20736 flip-flops, 41 B-SRAM and
#:   no multipliers, matching no device.
#: - Artix-7 XC7A35T: 20800 LUT6, 41600 flip-flops, 50 RAMB36 (counted as 100
#:   RAMB18 halves, so an 18 Kb block counts once), 90 DSP48E1 (Xilinx DS180).
_DEVICE_CAPACITY = {
    "ice40": {"luts": 5280, "ffs": 5280, "brams": 30, "dsps": 8},
    "ecp5": {"luts": 24288, "ffs": 24288, "brams": 56, "dsps": 28},
    "gowin": {"luts": 20736, "ffs": 15552, "brams": 46, "dsps": 48},
    "xilinx": {"luts": 20800, "ffs": 41600, "brams": 100, "dsps": 90},
}

_CAPACITY_DEVICE = {
    "ice40": "iCE40 UP5K",
    "ecp5": "ECP5 LFE5U-25F",
    "gowin": "Gowin GW2A-18",
    "xilinx": "Artix-7 XC7A35T",
}


_CAPACITY_KEYS = ("luts", "ffs", "brams", "dsps")

#: Seconds Yosys may run for one synthesis before the Studio stops it.
SYNTHESIS_TIMEOUT_SECONDS = 60

_NO_COST: dict[str, int] = {}

#: What each primitive a family's Yosys flow emits takes of the judged
#: resources. I/O buffers, constants, clock buffers, carry cells that ride in
#: a LUT's logic cell and wide-function multiplexers take none. A cell this
#: table does not know (a family's distributed RAM, a hand-placed primitive)
#: is reported and leaves the design without a fit verdict unless the counted
#: cells already overflow: its cost is unknown, so the counts are a floor.
#: Names were checked against Yosys 0.33 output for each family; the substring
#: matching it replaces counted Gowin wide multiplexers as LUTs, Gowin and
#: Xilinx LUT-RAM as block RAM, and missed ECP5 carry LUTs and DP16KD.
_CELL_COST: dict[str, dict[str, dict[str, int]]] = {
    "ice40": {
        "SB_LUT4": {"luts": 1},
        "SB_CARRY": _NO_COST,
        "SB_RAM40_4K": {"brams": 1},
        "SB_RAM40_4KNR": {"brams": 1},
        "SB_RAM40_4KNW": {"brams": 1},
        "SB_RAM40_4KNRNW": {"brams": 1},
        "SB_MAC16": {"dsps": 1},
        "SB_IO": _NO_COST,
        "SB_GB": _NO_COST,
        "SB_GB_IO": _NO_COST,
    },
    "ecp5": {
        "LUT4": {"luts": 1},
        "CCU2C": {"luts": 2},
        "PFUMX": _NO_COST,
        "L6MUX21": _NO_COST,
        "TRELLIS_FF": {"ffs": 1},
        "DP16KD": {"brams": 1},
        "PDPW16KD": {"brams": 1},
        "MULT18X18D": {"dsps": 1},
        "TRELLIS_IO": _NO_COST,
        "DCCA": _NO_COST,
        "VHI": _NO_COST,
        "VLO": _NO_COST,
    },
    "gowin": {
        **{f"LUT{width}": {"luts": 1} for width in range(1, 5)},
        "ALU": {"luts": 1},
        **{f"MUX2_LUT{width}": _NO_COST for width in range(5, 9)},
        **{name: {"brams": 1} for name in ("SP", "SPX9", "SDP", "SDPX9", "SDPB", "SDPX9B")},
        **{name: {"brams": 1} for name in ("DP", "DPX9", "DPB", "DPX9B", "ROM", "ROMX9")},
        **{name: {"brams": 1} for name in ("pROM", "pROMX9")},
        "MULT18X18": {"dsps": 1},
        "MULT36X36": {"dsps": 4},
        **{name: _NO_COST for name in ("IBUF", "OBUF", "IOBUF", "TBUF", "GND", "VCC")},
    },
    "xilinx": {
        **{f"LUT{width}": {"luts": 1} for width in range(1, 7)},
        "LUT6_2": {"luts": 1},
        "INV": {"luts": 1},
        "SRL16E": {"luts": 1},
        "SRLC32E": {"luts": 1},
        # Distributed RAM, in LUTs per primitive (Xilinx UG474).
        "RAM32X1S": {"luts": 1},
        "RAM64X1S": {"luts": 1},
        "RAM128X1S": {"luts": 2},
        "RAM256X1S": {"luts": 4},
        "RAM32X1D": {"luts": 2},
        "RAM64X1D": {"luts": 2},
        "RAM128X1D": {"luts": 4},
        "RAM32M": {"luts": 4},
        "RAM64M": {"luts": 4},
        "CARRY4": _NO_COST,
        "MUXF7": _NO_COST,
        "MUXF8": _NO_COST,
        **{name: {"ffs": 1} for name in ("FDRE", "FDSE", "FDCE", "FDPE", "LDCE", "LDPE")},
        "RAMB18E1": {"brams": 1},
        "RAMB36E1": {"brams": 2},
        "DSP48E1": {"dsps": 1},
        **{name: _NO_COST for name in ("IBUF", "OBUF", "OBUFT", "IOBUF", "BUFG", "BUFGCTRL")},
        **{name: _NO_COST for name in ("GND", "VCC")},
    },
}


def _flip_flop_cost(target: str, cell_type: str) -> dict[str, int] | None:
    """Recognise a family's flip-flop variants, which the tables do not list."""
    if target == "ice40" and cell_type.startswith("SB_DFF"):
        return {"ffs": 1}
    if target == "gowin" and cell_type.startswith("DFF"):
        return {"ffs": 1}
    return None


def cell_cost(target: str, cell_type: str) -> dict[str, int] | None:
    """Return what one cell takes of the judged resources, or None if unknown.

    Parameters
    ----------
    target : str
        Target identifier.
    cell_type : str
        The Yosys cell type.

    Returns
    -------
    dict[str, int] or None
        Resource amounts, empty for a cell that takes none, None for a cell
        whose cost the Studio does not know.
    """
    known = _CELL_COST.get(target, {}).get(cell_type)
    return known if known is not None else _flip_flop_cost(target, cell_type)


def capacity_verdict(
    resources: Mapping[str, Any],
    capacity: Mapping[str, int],
    *,
    device: str | None = None,
    uncounted: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Say whether a synthesised design fits its target device.

    Synthesis succeeding says the netlist exists, not that the device can
    hold it: a 20-neuron network synthesised to 6237 LUTs for a 5280-LUT
    UP5K and the pipeline reported it complete. Only resources the target's
    capacity lists are judged; a target without capacity data gets no
    verdict rather than a guessed one. Fitting by count is necessary, not
    sufficient: on iCE40 and ECP5 a LUT and a flip-flop share a logic cell
    (708 LUTs and 98 flip-flops took 752 UP5K cells), and placement and
    routing can still fail.

    Parameters
    ----------
    resources : Mapping[str, Any]
        Counted resources of the design.
    capacity : Mapping[str, int]
        The device's capacity per resource.
    device : str or None, optional
        The device the capacity describes, named with the verdict.
    uncounted : Mapping[str, int] or None, optional
        Cells whose cost is unknown, by type. With any, the counts are a
        floor: they can prove a design does not fit, never that it does.

    Returns
    -------
    dict[str, Any]
        ``fits_device`` (True, False, or None when unknown cells leave it
        open), ``exceeds_capacity`` (per resource, what the design needs and
        the device has), ``capacity_device`` and ``uncounted_cells``; an empty
        dict without capacity data.
    """
    if not capacity:
        return {}
    exceeds = {
        key: {"needed": int(resources.get(key, 0)), "available": int(capacity[key])}
        for key in _CAPACITY_KEYS
        if key in capacity and int(resources.get(key, 0)) > int(capacity[key])
    }
    unknown = dict(uncounted or {})
    fits: bool | None = False if exceeds else (None if unknown else True)
    return {
        "fits_device": fits,
        "exceeds_capacity": exceeds,
        "capacity_device": device,
        "uncounted_cells": unknown,
    }


def capacity_device_name(target: str) -> str:
    """Name the device a target's capacity is judged against.

    Parameters
    ----------
    target : str
        Target identifier.

    Returns
    -------
    str
        The device, or the target in capitals when none is named.
    """
    return _CAPACITY_DEVICE.get(target, target.upper())


def capacity_sentence(target: str, exceeds: Mapping[str, Mapping[str, int]]) -> str:
    """Say in words which resources a design needs beyond its device.

    Parameters
    ----------
    target : str
        Target identifier; its judged device is named.
    exceeds : Mapping[str, Mapping[str, int]]
        From :func:`capacity_verdict`.

    Returns
    -------
    str
        One sentence.
    """
    names = {"luts": "LUTs", "ffs": "flip-flops", "brams": "block RAMs", "dsps": "DSP blocks"}
    where = capacity_device_name(target)
    parts = [
        f"{row['needed']} {names.get(key, key)} (the device has {row['available']})"
        for key, row in exceeds.items()
    ]
    return f"the design needs {', '.join(parts)}: it does not fit the {where}"


def supported_targets() -> tuple[str, ...]:
    """Return synthesis targets accepted by the Studio EDA routes."""

    return tuple(_TARGETS)


def run_synthesis(
    verilog_source: str,
    target: str = "ice40",
    *,
    process_limits: EdaProcessLimits | None = None,
    tool_status: ToolStatusMap | None = None,
) -> dict[str, Any]:
    """Run Yosys synthesis and return resource usage.

    Parameters
    ----------
    verilog_source:
        SystemVerilog or Verilog source text to synthesise.
    target:
        Studio synthesis target identifier.
    process_limits:
        Optional host-supported CPU and address-space ceilings for the Yosys
        child process.
    tool_status:
        Optional path-free EDA tool status snapshot. When omitted, the backend
        captures a fresh snapshot for this result.

    Returns
    -------
    dict[str, Any]
        Path-free synthesis result with success state, target, resource counts,
        capacity metadata, utilisation, or a bounded error message.
    """
    if not isinstance(verilog_source, str):
        raise ValueError("verilog_source must be a string")
    if not verilog_source.strip():
        raise ValueError("verilog_source must not be empty")
    if len(verilog_source.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError("verilog_source exceeds 2 MiB size limit")
    if target not in _TARGETS:
        raise ValueError(f"Unknown target: {target}. Supported: {list(_TARGETS.keys())}")
    status = check_tools() if tool_status is None else tool_status
    target_provenance = build_synthesis_target_provenance(
        target,
        target_config=_TARGETS[target],
        capacity=_DEVICE_CAPACITY.get(target, {}),
        tool_status=status,
    ).to_public_dict()

    with tempfile.TemporaryDirectory(prefix="sc_synth_") as tmpdir:
        result, _json_path = _run_synthesis_in_directory(
            verilog_source,
            target,
            root=Path(tmpdir),
            process_limits=process_limits,
            target_provenance=target_provenance,
        )
        return result


def run_synthesis_terminal(
    verilog_source: str,
    target: str,
    *,
    compile_traceability: Mapping[str, object],
    cosim_parity: Mapping[str, object],
    process_limits: EdaProcessLimits | None = None,
) -> SynthesisTerminalExecution:
    """Run digest-bound synthesis and PnR for one parity-verified model RTL source."""

    if not isinstance(verilog_source, str) or not verilog_source.strip():
        raise ValueError("verilog_source must be a non-empty string")
    if len(verilog_source.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError("verilog_source exceeds 2 MiB size limit")
    if target not in _TARGETS:
        raise ValueError(f"Unknown target: {target}. Supported: {list(_TARGETS.keys())}")
    if _TARGETS[target]["pnr"] is None:
        raise ValueError(f"Target {target!r} has no place-and-route terminal.")

    source_chain = _validate_selected_rtl_chain(
        verilog_source,
        compile_traceability=compile_traceability,
        cosim_parity=cosim_parity,
    )
    tool_status = check_tools()
    target_provenance = build_synthesis_target_provenance(
        target,
        target_config=_TARGETS[target],
        capacity=_DEVICE_CAPACITY.get(target, {}),
        tool_status=tool_status,
    ).to_public_dict()

    with tempfile.TemporaryDirectory(prefix="sc_silicon_terminal_") as tmpdir:
        root = Path(tmpdir)
        synthesis, json_path = _run_synthesis_in_directory(
            verilog_source,
            target,
            root=root,
            process_limits=process_limits,
            target_provenance=target_provenance,
        )
        if not synthesis["success"] or json_path is None:
            return SynthesisTerminalExecution(
                netlist_json=None,
                report=_terminal_report(
                    source_chain=source_chain,
                    target=target,
                    target_provenance=target_provenance,
                    synthesis=synthesis,
                    pnr=None,
                    netlist_sha256=None,
                    routed_design_sha256=None,
                ),
                routed_design=None,
            )

        pnr = run_pnr(str(json_path), target, process_limits=process_limits)
        netlist_json = json_path.read_bytes()
        routed_path = _pnr_output_path(json_path, target)
        routed_design = None
        if routed_path.is_file():
            if routed_path.stat().st_size > _MAX_TERMINAL_ARTIFACT_BYTES:
                pnr = {
                    "success": False,
                    "error": "Routed design exceeds 16 MiB artifact limit",
                }
            else:
                routed_design = routed_path.read_bytes()
        elif pnr.get("success") is True:
            pnr = {
                "success": False,
                "error": "Place-and-route completed without a routed-design artifact",
            }
        return SynthesisTerminalExecution(
            netlist_json=netlist_json,
            report=_terminal_report(
                source_chain=source_chain,
                target=target,
                target_provenance=target_provenance,
                synthesis=synthesis,
                pnr=pnr,
                netlist_sha256=_sha256_bytes(netlist_json),
                routed_design_sha256=(
                    _sha256_bytes(routed_design) if routed_design is not None else None
                ),
            ),
            routed_design=routed_design,
        )


def _run_synthesis_in_directory(
    verilog_source: str,
    target: str,
    *,
    root: Path,
    process_limits: EdaProcessLimits | None,
    target_provenance: Mapping[str, object],
) -> tuple[dict[str, Any], Path | None]:
    """Run Yosys in one trusted directory and retain its netlist for a caller."""

    v_path = root / "design.v"
    json_path = root / "design.json"
    log_path = root / "yosys.log"
    script_path = root / "synth.ys"
    v_path.write_text(verilog_source, encoding="utf-8")
    synth_cmd = _TARGETS[target]["synth_cmd"]
    script_path.write_text(
        # write_json and not "-json": synth_xilinx has no such option, so
        # the Xilinx target failed on every design; synth_gowin's "-json"
        # also withholds block RAM for nextpnr-gowin, which this target does
        # not run, and built a 256x16 memory from 64 LUT-RAMs. For iCE40 and
        # ECP5 the two are the same command.
        f"read_verilog {v_path.name}; {synth_cmd}; write_json {json_path.name}",
        encoding="utf-8",
    )

    yosys_executable = _resolve_eda_tool("yosys")
    if yosys_executable is None:
        return (
            {
                "success": False,
                "error": "yosys not found. Install: https://github.com/YosysHQ/yosys",
                "target": target,
                "target_provenance": dict(target_provenance),
            },
            None,
        )
    try:
        completed = _run_eda_command(
            [yosys_executable, "-s", str(script_path)],
            cwd=root,
            timeout_seconds=SYNTHESIS_TIMEOUT_SECONDS,
            process_limits=process_limits,
        )
        log = completed.stdout + completed.stderr
        log_path.write_text(log, encoding="utf-8")
    except FileNotFoundError:
        return (
            {
                "success": False,
                "error": "yosys not found",
                "target": target,
                "target_provenance": dict(target_provenance),
            },
            None,
        )
    except subprocess.TimeoutExpired:
        return (
            {
                "success": False,
                "error": f"Synthesis timed out ({SYNTHESIS_TIMEOUT_SECONDS}s)",
                "timed_out": True,
                "target": target,
                "target_provenance": dict(target_provenance),
            },
            None,
        )
    if not json_path.exists():
        return (
            {
                "success": False,
                "error": _yosys_failure_message(log),
                "target": target,
                "target_provenance": dict(target_provenance),
            },
            None,
        )

    resources, uncounted = _parse_yosys_json(str(json_path), target)
    capacity = _DEVICE_CAPACITY.get(target, {})
    return (
        {
            "success": True,
            "target": target,
            "resources": resources,
            "capacity": capacity,
            "utilisation": {
                key: round(resources.get(key, 0) / max(capacity.get(key, 1), 1) * 100, 1)
                for key in ["luts", "ffs", "brams", "dsps"]
            },
            **capacity_verdict(
                resources, capacity, device=_CAPACITY_DEVICE.get(target), uncounted=uncounted
            ),
            "log_excerpt": log[-300:] if log else "",
            "target_provenance": dict(target_provenance),
        },
        json_path,
    )


def _yosys_failure_message(log: str) -> str:
    """Say why Yosys wrote no netlist, from its own error lines.

    Yosys prints a pass's whole help text after a command error, so the last
    500 characters of the log, which this message used to be, showed the help
    and not the error. Its standard output can end mid-line, so an error line
    from standard error may follow other text on the same line.

    Parameters
    ----------
    log : str
        Yosys's standard output and error.

    Returns
    -------
    str
        The error lines, or the end of the log when it has none.
    """
    errors = [line[line.index("ERROR:") :].strip() for line in log.splitlines() if "ERROR:" in line]
    if errors:
        return "Synthesis failed: " + "; ".join(errors[:3])
    return f"Synthesis failed. Log:\n{log[-500:]}"


def _validate_selected_rtl_chain(
    verilog_source: str,
    *,
    compile_traceability: Mapping[str, object],
    cosim_parity: Mapping[str, object],
) -> dict[str, object]:
    """Validate selected-model compile and bit-exact parity evidence against RTL bytes."""

    actual_rtl_sha256 = _sha256_bytes(verilog_source.encode("utf-8"))
    if compile_traceability.get("schema_version") != "studio.compile-traceability.v1":
        raise ValueError("Selected RTL terminal requires studio.compile-traceability.v1 evidence.")
    if compile_traceability.get("source") != "model":
        raise ValueError("Selected RTL terminal requires catalogue-model compile evidence.")
    if compile_traceability.get("status") != "completed":
        raise ValueError("Selected RTL compile evidence is not completed.")
    output = compile_traceability.get("output")
    source_payload = compile_traceability.get("source_payload")
    if not isinstance(output, Mapping) or not isinstance(source_payload, Mapping):
        raise ValueError("Selected RTL compile evidence is malformed.")
    if output.get("rtl_sha256") != actual_rtl_sha256:
        raise ValueError("Selected RTL does not match the compile output digest.")
    digest_source_payload = _browser_stable_model_source_payload(source_payload)
    if compile_traceability.get("input_sha256") != _sha256_json(digest_source_payload):
        raise ValueError("Selected RTL compile input digest is invalid.")
    trace_payload = dict(compile_traceability)
    claimed_trace_sha256 = trace_payload.pop("traceability_sha256", None)
    trace_payload["source_payload"] = digest_source_payload
    if claimed_trace_sha256 != _sha256_json(trace_payload):
        raise ValueError("Selected RTL compile traceability digest is invalid.")

    if cosim_parity.get("schema_version") != "studio.cosim-parity.v1":
        raise ValueError("Selected RTL terminal requires studio.cosim-parity.v1 evidence.")
    if cosim_parity.get("status") != "completed" or cosim_parity.get("bit_exact") is not True:
        raise ValueError("Selected RTL co-simulation is not bit-exact and completed.")
    rtl = cosim_parity.get("rtl")
    configuration = cosim_parity.get("configuration")
    if not isinstance(rtl, Mapping) or not isinstance(configuration, Mapping):
        raise ValueError("Selected RTL co-simulation evidence is malformed.")
    if rtl.get("source_sha256") != actual_rtl_sha256:
        raise ValueError("Selected RTL does not match the co-simulated source digest.")
    for key in ("dt", "integrator", "model_name", "q_format", "schema_name", "schema_sha256"):
        if configuration.get(key) != source_payload.get(key):
            raise ValueError(f"Selected RTL compile/co-simulation field {key!r} does not match.")
    if cosim_parity.get("module_name") != output.get("module_name"):
        raise ValueError("Selected RTL compile/co-simulation module does not match.")

    return {
        "compile_input_sha256": compile_traceability["input_sha256"],
        "compile_traceability_sha256": claimed_trace_sha256,
        "cosim_reference_trace_sha256": _nested_string(cosim_parity, "reference", "trace_sha256"),
        "cosim_rtl_trace_sha256": _nested_string(cosim_parity, "rtl", "trace_sha256"),
        "model_name": source_payload.get("model_name"),
        "module_name": output.get("module_name"),
        "rtl_sha256": actual_rtl_sha256,
    }


def _browser_stable_model_source_payload(
    source_payload: Mapping[str, object],
) -> dict[str, object]:
    """Restore model float fields after a standards-compliant browser JSON round trip."""

    dt = source_payload.get("dt")
    params = source_payload.get("params")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise ValueError("Selected RTL compile source dt is invalid.")
    if not isinstance(params, Mapping):
        raise ValueError("Selected RTL compile source parameters are invalid.")
    canonical_params: dict[str, float] = {}
    for key, value in params.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError("Selected RTL compile source parameters are invalid.")
        canonical_params[key] = float(value)
    return {
        **source_payload,
        "dt": float(dt),
        "params": canonical_params,
    }


def _nested_string(payload: Mapping[str, object], outer: str, inner: str) -> str:
    nested = payload.get(outer)
    value = nested.get(inner) if isinstance(nested, Mapping) else None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Selected RTL evidence digest {outer}.{inner} is invalid.")
    return value


def _terminal_report(
    *,
    source_chain: Mapping[str, object],
    target: str,
    target_provenance: Mapping[str, object],
    synthesis: Mapping[str, object],
    pnr: Mapping[str, object] | None,
    netlist_sha256: str | None,
    routed_design_sha256: str | None,
) -> dict[str, Any]:
    success = bool(synthesis.get("success")) and pnr is not None and bool(pnr.get("success"))
    return {
        "artifacts": {
            "netlist_sha256": netlist_sha256,
            "routed_design_sha256": routed_design_sha256,
        },
        "evidence_classification": "synthesis",
        "place_and_route": dict(pnr) if pnr is not None else None,
        "schema_version": "studio.silicon-terminal.v1",
        "source_chain": dict(source_chain),
        "status": "completed" if success else "failed",
        "success": success,
        "synthesis": dict(synthesis),
        "target": target,
        "target_provenance": dict(target_provenance),
    }


def _sha256_json(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_yosys_json(json_path: str, target: str) -> tuple[dict[str, int], dict[str, int]]:
    """Count a synthesised design's resources from Yosys JSON output.

    Only the design's own modules are read: Yosys also writes the family's
    cell library (blackbox and whitebox modules holding timing cells), which
    counted towards "cells" and "wires".

    Parameters
    ----------
    json_path : str
        The netlist Yosys wrote.
    target : str
        Target identifier, selecting the family's cell table.

    Returns
    -------
    tuple[dict[str, int], dict[str, int]]
        The counted resources, and the cells whose cost is unknown by type.
    """
    with open(json_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid Yosys JSON payload: expected top-level object")
    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("Invalid Yosys JSON payload: 'modules' must be an object")

    resources = {"luts": 0, "ffs": 0, "brams": 0, "dsps": 0, "cells": 0, "wires": 0}
    uncounted: dict[str, int] = {}

    for mod_name, mod in modules.items():
        if not isinstance(mod, dict):
            raise ValueError(f"Invalid Yosys JSON payload: module '{mod_name}' must be an object")
        attributes = mod.get("attributes", {})
        if isinstance(attributes, dict) and ("blackbox" in attributes or "whitebox" in attributes):
            continue
        cells = mod.get("cells", {})
        if not isinstance(cells, dict):
            raise ValueError(
                f"Invalid Yosys JSON payload: module '{mod_name}.cells' must be an object"
            )
        resources["cells"] += len(cells)
        for cell_name, cell in cells.items():
            if not isinstance(cell, dict):
                raise ValueError(
                    f"Invalid Yosys JSON payload: module '{mod_name}.cells.{cell_name}' must be an object"
                )
            cell_type = str(cell.get("type", ""))
            cost = cell_cost(target, cell_type)
            if cost is None:
                uncounted[cell_type] = uncounted.get(cell_type, 0) + 1
                continue
            for key, amount in cost.items():
                resources[key] += amount
        netnames = mod.get("netnames", {})
        if not isinstance(netnames, dict):
            raise ValueError(
                f"Invalid Yosys JSON payload: module '{mod_name}.netnames' must be an object"
            )
        resources["wires"] += len(netnames)

    return resources, uncounted


def estimate_resources(ir_op_count: int, target: str = "ice40") -> dict[str, Any]:
    """Quick resource estimate from IR operation count, no Yosys needed.

    Heuristic: each IR op maps to ~2 LUTs + 1 FF on average.
    LIF step op maps to ~12 LUTs + 8 FFs + 1 DSP (multiplier).
    """
    if target not in _TARGETS:
        raise ValueError(f"Unknown target: {target}. Supported: {list(_TARGETS.keys())}")
    capacity = _DEVICE_CAPACITY.get(target, _DEVICE_CAPACITY["ice40"])
    est_luts = ir_op_count * 2 + 12
    est_ffs = ir_op_count + 8
    est_dsps = 1
    est_brams = 0
    resources = {"luts": est_luts, "ffs": est_ffs, "brams": est_brams, "dsps": est_dsps}
    return {
        "target": target,
        "estimated": True,
        "resources": resources,
        "capacity": capacity,
        "utilisation": {
            k: round(resources[k] / max(capacity.get(k, 1), 1) * 100, 1)
            for k in ["luts", "ffs", "brams", "dsps"]
        },
        **capacity_verdict(resources, capacity, device=_CAPACITY_DEVICE.get(target)),
    }


def multi_target_synthesis(
    verilog_source: str,
    *,
    process_limits: EdaProcessLimits | None = None,
) -> dict[str, Any]:
    """Run synthesis on all supported targets and return a comparison.

    Parameters
    ----------
    verilog_source:
        SystemVerilog or Verilog source text to synthesise.
    process_limits:
        Optional host-supported CPU and address-space ceilings applied to every
        Yosys child process.

    Returns
    -------
    dict[str, Any]
        Mapping with per-target synthesis results and the supported target list.
    """
    if not isinstance(verilog_source, str):
        raise ValueError("verilog_source must be a string")
    if not verilog_source.strip():
        raise ValueError("verilog_source must not be empty")
    if len(verilog_source.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError("verilog_source exceeds 2 MiB size limit")
    tool_status = check_tools()
    results = {}
    for target in _TARGETS:
        results[target] = run_synthesis(
            verilog_source,
            target,
            process_limits=process_limits,
            tool_status=tool_status,
        )
    return {
        "target_provenance_matrix": build_synthesis_target_provenance_matrix(
            targets=_TARGETS,
            capacities=_DEVICE_CAPACITY,
            tool_status=tool_status,
        ),
        "targets": results,
        "supported": list(_TARGETS.keys()),
    }


def run_pnr(
    json_path: str,
    target: str = "ice40",
    *,
    process_limits: EdaProcessLimits | None = None,
) -> dict[str, Any]:
    """Run nextpnr place-and-route and return timing report.

    Parameters
    ----------
    json_path:
        Path to a Yosys JSON netlist. The path must point to a regular JSON
        file and must not be a symlink.
    target:
        Studio target identifier with nextpnr support.
    process_limits:
        Optional host-supported CPU and address-space ceilings for the nextpnr
        child process.

    Returns
    -------
    dict[str, Any]
        Path-free PnR result with success state, timing metadata, log excerpt,
        or a bounded error message.
    """
    cfg = _TARGETS.get(target)
    if not cfg or not cfg["pnr"]:
        return {"success": False, "error": f"No PnR tool for target {target}"}

    raw_json_path = Path(json_path).expanduser()
    if raw_json_path.suffix.lower() != ".json":
        return {"success": False, "error": "PnR input must be a .json netlist file"}
    if raw_json_path.is_symlink():
        return {"success": False, "error": f"PnR input must not be a symlink: {raw_json_path}"}
    resolved_json = raw_json_path.resolve()
    if not resolved_json.exists():
        return {"success": False, "error": f"PnR input does not exist: {resolved_json}"}
    if not resolved_json.is_file():
        return {"success": False, "error": f"PnR input is not a regular file: {resolved_json}"}
    if resolved_json.stat().st_size > _MAX_TERMINAL_ARTIFACT_BYTES:
        return {"success": False, "error": "PnR input exceeds 16 MiB size limit"}
    try:
        with resolved_json.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"success": False, "error": "PnR input is not valid UTF-8 JSON"}
    if not isinstance(payload, dict):
        return {"success": False, "error": "PnR input JSON must be an object"}

    output_path = _pnr_output_path(resolved_json, target)
    pnr_tool = cfg["pnr"]
    if pnr_tool is None:
        return {"success": False, "error": f"No PnR tool for target {target}"}
    pnr_executable = _resolve_eda_tool(pnr_tool)
    if pnr_executable is None:
        return {"success": False, "error": f"{pnr_tool} not found"}

    try:
        result = _run_eda_command(
            _pnr_command(
                executable=pnr_executable,
                config=cfg,
                json_path=resolved_json,
                output_path=output_path,
            ),
            timeout_seconds=120,
            process_limits=process_limits,
        )
        log = result.stdout + result.stderr

        max_freq = None
        critical_path = None
        for line in log.split("\n"):
            if "Max frequency" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        max_freq = float(parts[-1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
            if "critical path" in line.lower():
                critical_path = line.strip()

        return {
            "success": result.returncode == 0,
            "max_freq_mhz": max_freq,
            "critical_path": critical_path,
            "log_excerpt": log[-300:],
        }
    except FileNotFoundError:
        return {"success": False, "error": f"{pnr_tool} not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "PnR timed out (120s)"}


def _pnr_output_path(json_path: Path, target: str) -> Path:
    """Return the target-native routed-design artifact path."""

    suffix = ".config" if target == "ecp5" else ".asc"
    return json_path.with_suffix(suffix)


def _pnr_command(
    *,
    executable: str,
    config: Mapping[str, str | None],
    json_path: Path,
    output_path: Path,
) -> list[str]:
    """Build a target-correct nextpnr command without accepting client flags."""

    device = config.get("device")
    if device is None:
        raise ValueError("PnR target device is not configured.")
    command = [executable, f"--{device}", "--json", str(json_path)]
    package = config.get("package")
    if package is not None:
        command.extend(["--package", package])
    output_flag = "--textcfg" if config.get("pnr") == "nextpnr-ecp5" else "--asc"
    command.extend([output_flag, str(output_path)])
    return command
