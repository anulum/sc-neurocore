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
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        if limits.address_space_bytes is not None and hasattr(resource, "RLIMIT_AS"):
            memory_limit = int(limits.address_space_bytes)
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
        f"read_verilog {v_path.name}; {synth_cmd} -json {json_path.name}",
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
            timeout_seconds=60,
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
                "error": "Synthesis timed out (60s)",
                "target": target,
                "target_provenance": dict(target_provenance),
            },
            None,
        )
    if not json_path.exists():
        return (
            {
                "success": False,
                "error": f"Synthesis failed. Log:\n{log[-500:]}",
                "target": target,
                "target_provenance": dict(target_provenance),
            },
            None,
        )

    resources = _parse_yosys_json(str(json_path))
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
            "log_excerpt": log[-300:] if log else "",
            "target_provenance": dict(target_provenance),
        },
        json_path,
    )


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
            ctype = str(cell.get("type", "")).upper()
            if "LUT" in ctype:
                resources["luts"] += 1
            elif _is_flip_flop_cell(ctype):
                resources["ffs"] += 1
            elif "RAM" in ctype:
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


def _is_flip_flop_cell(cell_type: str) -> bool:
    """Recognise Yosys flip-flop cells across supported target libraries."""

    return (
        "DFF" in cell_type
        or cell_type.endswith("_FF")
        or cell_type in {"FDCE", "FDPE", "FDRE", "FDSE"}
    )


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
