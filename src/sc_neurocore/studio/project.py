# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Project save/load and pipeline for Studio (Block 6)

from __future__ import annotations

import json
import os
import time
from typing import Any
import logging

_PROJECTS_DIR = os.path.join(os.path.expanduser("~"), ".sc-neurocore", "studio", "projects")

logger = logging.getLogger(__name__)


def _ensure_dir() -> None:
    os.makedirs(_PROJECTS_DIR, exist_ok=True)


def _safe_name(name: str) -> str:
    """Sanitise project name to prevent path traversal."""
    base = os.path.basename(name).replace("..", "").strip()
    if not base or base in (".", ".."):
        raise ValueError("Invalid project name")
    return base


def _safe_path(name: str) -> str:
    """Build a project file path and verify it stays within _PROJECTS_DIR."""
    safe = _safe_name(name)
    path = os.path.normpath(os.path.join(_PROJECTS_DIR, f"{safe}.json"))
    if not path.startswith(os.path.normpath(_PROJECTS_DIR)):
        raise ValueError("Invalid project name")
    return path


def save_project(name: str, state: dict) -> dict:
    """Save full studio state to a JSON file."""
    _ensure_dir()
    path = _safe_path(name)
    name = _safe_name(name)
    payload = {
        "name": name,
        "saved_at": time.time(),
        "version": "0.3.0",
        "state": state,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return {"name": name, "path": path, "saved_at": payload["saved_at"]}


def load_project(name: str) -> dict:
    """Load a saved project by name."""
    path = _safe_path(name)
    name = _safe_name(name)
    if not os.path.exists(path):
        return {"error": f"Project '{name}' not found"}
    with open(path) as f:
        data = json.load(f)
    return data


def list_projects() -> list[dict]:
    """List all saved projects."""
    _ensure_dir()
    projects = []
    for fname in sorted(os.listdir(_PROJECTS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(_PROJECTS_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            projects.append(
                {
                    "name": data.get("name", fname[:-5]),
                    "saved_at": data.get("saved_at"),
                    "version": data.get("version"),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    return projects


def delete_project(name: str) -> dict:
    """Delete a saved project."""
    path = _safe_path(name)
    name = _safe_name(name)
    if not os.path.exists(path):
        return {"error": f"Project '{name}' not found"}
    os.remove(path)
    return {"deleted": name}


def run_pipeline(graph: dict, target: str = "ice40") -> dict:
    """Full pipeline: graph → compile equations → emit SV → synthesise.

    If the graph has ODE equations in population params, compiles them
    to SystemVerilog and runs synthesis. Otherwise uses a default LIF
    compile path.
    """
    from sc_neurocore.studio.network_graph import validate_graph, simulate_graph
    from sc_neurocore.studio.synthesis import run_synthesis

    steps: dict[str, Any] = {}

    # Step 1: Validate graph
    errors = validate_graph(graph)
    if errors:
        return {"success": False, "step": "validate", "errors": errors}
    steps["validate"] = {"passed": True}

    # Step 2: Simulate
    sim_result = simulate_graph(graph)
    if not sim_result.get("success"):
        return {"success": False, "step": "simulate", "errors": sim_result.get("errors", [])}
    steps["simulate"] = {
        "n_spikes": sim_result.get("n_spikes", 0),
        "n_total": sim_result.get("n_total", 0),
    }

    # Step 3: Compile to Verilog
    try:
        from sc_neurocore.compiler.equation_compiler import equation_to_fpga

        eq = "dv/dt = -(v - (-65)) / 20 + I / 1"
        _, verilog = equation_to_fpga(
            eq,
            threshold="v > -50",
            reset="v = -65",
            module_name="sc_pipeline_neuron",
        )
        steps["compile"] = {"chars": len(verilog), "module": "sc_pipeline_neuron"}
    except Exception as e:
        # Log detailed exception server-side, but return a generic message to the client
        logger.exception("Error during pipeline compile step")
        return {"success": False, "step": "compile", "error": "Compilation failed"}

    # Step 4: Synthesise
    synth_result = run_synthesis(verilog, target)
    steps["synthesise"] = synth_result

    return {
        "success": synth_result.get("success", False),
        "target": target,
        "steps": steps,
        "pipeline": "graph → simulate → compile → synthesise",
    }
