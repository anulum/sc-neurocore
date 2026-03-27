# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synthesis Dashboard backend for Studio

from __future__ import annotations

import json
import os
import subprocess
import tempfile


def check_tools() -> dict:
    """Detect which EDA tools are installed."""
    tools = {}
    for name, cmd in [
        ("yosys", ["yosys", "--version"]),
        ("nextpnr_ice40", ["nextpnr-ice40", "--version"]),
        ("nextpnr_ecp5", ["nextpnr-ecp5", "--version"]),
        ("firtool", ["firtool", "--version"]),
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            version = r.stdout.strip().split("\n")[0] if r.returncode == 0 else None
            tools[name] = {"available": r.returncode == 0, "version": version}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            tools[name] = {"available": False, "version": None}
    return tools


_TARGETS = {
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


def run_synthesis(verilog_source: str, target: str = "ice40") -> dict:
    """Run Yosys synthesis and return resource usage."""
    if target not in _TARGETS:
        raise ValueError(f"Unknown target: {target}. Supported: {list(_TARGETS.keys())}")

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

        try:
            result = subprocess.run(
                ["yosys", "-s", script_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            log = result.stdout + result.stderr
            with open(log_path, "w") as f:
                f.write(log)
        except FileNotFoundError:
            return {
                "success": False,
                "error": "yosys not found. Install: https://github.com/YosysHQ/yosys",
                "target": target,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Synthesis timed out (60s)", "target": target}

        if not os.path.exists(json_path):
            return {
                "success": False,
                "error": f"Synthesis failed. Log:\n{log[-500:]}",
                "target": target,
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
        }


def _parse_yosys_json(json_path: str) -> dict:
    """Extract resource counts from Yosys JSON output."""
    with open(json_path) as f:
        data = json.load(f)

    resources = {"luts": 0, "ffs": 0, "brams": 0, "dsps": 0, "cells": 0, "wires": 0}

    for mod_name, mod in data.get("modules", {}).items():
        cells = mod.get("cells", {})
        resources["cells"] += len(cells)
        for cell_name, cell in cells.items():
            ctype = cell.get("type", "")
            if "LUT" in ctype or "SB_LUT" in ctype:
                resources["luts"] += 1
            elif "DFF" in ctype or "SB_DFF" in ctype:
                resources["ffs"] += 1
            elif "RAM" in ctype or "BRAM" in ctype or "SB_RAM" in ctype:
                resources["brams"] += 1
            elif "DSP" in ctype or "MUL" in ctype:
                resources["dsps"] += 1
        resources["wires"] += len(mod.get("netnames", {}))

    return resources


def run_pnr(json_path: str, target: str = "ice40") -> dict:
    """Run nextpnr place-and-route and return timing report."""
    cfg = _TARGETS.get(target)
    if not cfg or not cfg["pnr"]:
        return {"success": False, "error": f"No PnR tool for target {target}"}

    asc_path = json_path.replace(".json", ".asc")
    try:
        result = subprocess.run(
            [cfg["pnr"], f"--{cfg['device']}", "--json", json_path, "--asc", asc_path],
            capture_output=True,
            text=True,
            timeout=120,
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
        return {"success": False, "error": f"{cfg['pnr']} not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "PnR timed out (120s)"}
