# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synthesis Dashboard backend for Studio

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        if limits.address_space_bytes is not None and hasattr(resource, "RLIMIT_AS"):
            memory_limit = int(limits.address_space_bytes)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

    return apply_limits


def _run_eda_command(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    process_limits: EdaProcessLimits | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one allowlisted EDA command with optional child-process limits."""

    limit_preexec = _build_limit_preexec(process_limits)
    if limit_preexec is None:
        return subprocess.run(  # nosec B603
            list(command),
            capture_output=True,
            shell=False,
            text=True,
            timeout=timeout_seconds,
        )
    return subprocess.run(  # nosec B603
        list(command),
        capture_output=True,
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
            version = r.stdout.strip().split("\n")[0] if r.returncode == 0 else None
            tools[name] = {"available": r.returncode == 0, "version": version}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            tools[name] = {"available": False, "version": None}
    return tools


_TARGETS: dict[str, dict[str, str | None]] = {
    "ice40": {"synth_cmd": "synth_ice40", "pnr": "nextpnr-ice40", "device": "up5k"},
    "ecp5": {"synth_cmd": "synth_ecp5", "pnr": "nextpnr-ecp5", "device": "25k"},
    "gowin": {"synth_cmd": "synth_gowin", "pnr": None, "device": None},
    "xilinx": {"synth_cmd": "synth_xilinx", "pnr": None, "device": None},
}

_DEVICE_CAPACITY = {
    "ice40": {"luts": 5280, "ffs": 5280, "brams": 30, "dsps": 0},
    "ecp5": {"luts": 24576, "ffs": 24576, "brams": 56, "dsps": 28},
    "gowin": {"luts": 20736, "ffs": 20736, "brams": 41, "dsps": 0},
    "xilinx": {"luts": 20800, "ffs": 41600, "brams": 50, "dsps": 90},
}


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
        v_path = os.path.join(tmpdir, "design.v")
        json_path = os.path.join(tmpdir, "design.json")
        log_path = os.path.join(tmpdir, "yosys.log")

        with open(v_path, "w") as f:
            f.write(verilog_source)

        synth_cmd = _TARGETS[target]["synth_cmd"]
        script = f"read_verilog {v_path}; {synth_cmd} -json {json_path}"
        script_path = os.path.join(tmpdir, "synth.ys")
        with open(script_path, "w") as f:
            f.write(script)

        yosys_executable = _resolve_eda_tool("yosys")
        if yosys_executable is None:
            return {
                "success": False,
                "error": "yosys not found. Install: https://github.com/YosysHQ/yosys",
                "target": target,
                "target_provenance": target_provenance,
            }

        try:
            result = _run_eda_command(
                [yosys_executable, "-s", script_path],
                timeout_seconds=60,
                process_limits=process_limits,
            )
            log = result.stdout + result.stderr
            with open(log_path, "w") as f:
                f.write(log)
        except FileNotFoundError:
            return {
                "success": False,
                "error": "yosys not found",
                "target": target,
                "target_provenance": target_provenance,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Synthesis timed out (60s)",
                "target": target,
                "target_provenance": target_provenance,
            }

        if not os.path.exists(json_path):
            return {
                "success": False,
                "error": f"Synthesis failed. Log:\n{log[-500:]}",
                "target": target,
                "target_provenance": target_provenance,
            }

        resources = _parse_yosys_json(json_path)
        capacity = _DEVICE_CAPACITY.get(target, {})

        return {
            "success": True,
            "target": target,
            "resources": resources,
            "capacity": capacity,
            "utilisation": {
                k: round(resources.get(k, 0) / max(capacity.get(k, 1), 1) * 100, 1)
                for k in ["luts", "ffs", "brams", "dsps"]
            },
            "log_excerpt": log[-300:] if log else "",
            "target_provenance": target_provenance,
        }


def _parse_yosys_json(json_path: str) -> dict[str, Any]:
    """Extract resource counts from Yosys JSON output."""
    with open(json_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid Yosys JSON payload: expected top-level object")
    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("Invalid Yosys JSON payload: 'modules' must be an object")

    resources = {"luts": 0, "ffs": 0, "brams": 0, "dsps": 0, "cells": 0, "wires": 0}

    for mod_name, mod in modules.items():
        if not isinstance(mod, dict):
            raise ValueError(f"Invalid Yosys JSON payload: module '{mod_name}' must be an object")
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
            ctype = cell.get("type", "")
            if "LUT" in ctype or "SB_LUT" in ctype:
                resources["luts"] += 1
            elif "DFF" in ctype or "SB_DFF" in ctype:
                resources["ffs"] += 1
            elif "RAM" in ctype or "BRAM" in ctype or "SB_RAM" in ctype:
                resources["brams"] += 1
            elif "DSP" in ctype or "MUL" in ctype:
                resources["dsps"] += 1
        netnames = mod.get("netnames", {})
        if not isinstance(netnames, dict):
            raise ValueError(
                f"Invalid Yosys JSON payload: module '{mod_name}.netnames' must be an object"
            )
        resources["wires"] += len(netnames)

    return resources


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
    if resolved_json.stat().st_size > 16 * 1024 * 1024:
        return {"success": False, "error": "PnR input exceeds 16 MiB size limit"}
    try:
        with resolved_json.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"success": False, "error": "PnR input is not valid UTF-8 JSON"}
    if not isinstance(payload, dict):
        return {"success": False, "error": "PnR input JSON must be an object"}

    asc_path = str(resolved_json.with_suffix(".asc"))
    pnr_tool = cfg["pnr"]
    if pnr_tool is None:
        return {"success": False, "error": f"No PnR tool for target {target}"}
    pnr_executable = _resolve_eda_tool(pnr_tool)
    if pnr_executable is None:
        return {"success": False, "error": f"{pnr_tool} not found"}

    try:
        result = _run_eda_command(
            [pnr_executable, f"--{cfg['device']}", "--json", str(resolved_json), "--asc", asc_path],
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
