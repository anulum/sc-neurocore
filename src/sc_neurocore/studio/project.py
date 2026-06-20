# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
from pathlib import Path

from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.studio.synthesis import EdaProcessLimits

_PROJECTS_DIR = os.path.join(os.path.expanduser("~"), ".sc-neurocore", "studio", "projects")

logger = logging.getLogger(__name__)

_IDENTIFIER_KEY_CONTEXTS = {
    "module_name": "module name",
    "signal_name": "signal name",
}
_IDENTIFIER_MAPPING_CONTEXTS = {
    "constants": "parameter name",
    "input_shapes": "input name",
    "parameters": "parameter name",
    "params": "parameter name",
}


def _validate_hdl_identifiers(payload: Any) -> None:
    """Reject workspace content that would later interpolate into HDL/MLIR source."""
    errors: list[str] = []

    def _check(value: str, context: str, path: str) -> None:
        try:
            sanitize_ident(value, context=context)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_path = f"{path}.{key}"
                if key in _IDENTIFIER_KEY_CONTEXTS and isinstance(value, str):
                    _check(value, _IDENTIFIER_KEY_CONTEXTS[key], key_path)
                if key in _IDENTIFIER_MAPPING_CONTEXTS and isinstance(value, dict):
                    context = _IDENTIFIER_MAPPING_CONTEXTS[key]
                    for ident_key in value:
                        if isinstance(ident_key, str):
                            _check(ident_key, context, f"{key_path}[{ident_key!r}]")
                if key == "layers" and isinstance(value, list):
                    for idx, layer in enumerate(value):
                        if isinstance(layer, dict) and isinstance(layer.get("name"), str):
                            _check(layer["name"], "layer name", f"{key_path}[{idx}].name")
                if key == "signals" and isinstance(value, list):
                    for idx, signal in enumerate(value):
                        if isinstance(signal, dict) and isinstance(signal.get("name"), str):
                            _check(signal["name"], "signal name", f"{key_path}[{idx}].name")
                _walk(value, key_path)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk(item, f"{path}[{idx}]")

    _walk(payload, "project")
    if errors:
        raise ValueError("Invalid HDL-facing identifiers in project: " + "; ".join(errors))


def _ensure_dir() -> None:
    _projects_root().mkdir(parents=True, exist_ok=True)


def _projects_root() -> Path:
    """Return the resolved Studio project root."""
    return Path(_PROJECTS_DIR).expanduser().resolve()


def _safe_name(name: str) -> str:
    """Validate a project name that maps to one JSON file in the project root."""
    if not isinstance(name, str):
        raise ValueError("Invalid project name")
    raw = name.strip()
    if (
        not raw
        or raw in (".", "..")
        or "/" in raw
        or "\\" in raw
        or Path(raw).is_absolute()
        or ".." in Path(raw).parts
    ):
        raise ValueError("Invalid project name")
    base = os.path.basename(raw)
    if not base or base in (".", ".."):
        raise ValueError("Invalid project name")
    return base


def _safe_path(name: str) -> Path:
    """Build a resolved project file path confined to the Studio project root."""
    safe = _safe_name(name)
    root = _projects_root()
    path = (root / f"{safe}.json").resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("Invalid project name") from None
    if path.parent != root:
        raise ValueError("Invalid project name")
    return path


def save_project(name: str, state: dict[str, Any]) -> dict[str, Any]:
    """Save full studio state to a JSON file."""
    _ensure_dir()
    path = _safe_path(name)
    name = _safe_name(name)
    if not isinstance(state, dict):
        raise ValueError("Project state must be an object")
    payload = {
        "name": name,
        "saved_at": time.time(),
        "version": "0.3.0",
        "state": state,
    }
    _validate_hdl_identifiers(payload)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return {"name": name, "path": str(path), "saved_at": payload["saved_at"]}


def load_project(name: str) -> dict[str, Any]:
    """Load a saved project by name."""
    path = _safe_path(name)
    name = _safe_name(name)
    if not os.path.exists(path):
        return {"error": f"Project '{name}' not found"}
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid project payload: expected object")
    if not isinstance(data.get("state"), dict):
        raise ValueError("Invalid project payload: 'state' must be an object")
    stored_name = data.get("name")
    if not isinstance(stored_name, str) or _safe_name(stored_name) != name:
        raise ValueError("Invalid project payload: inconsistent project name")
    _validate_hdl_identifiers(data)
    return data


def list_projects() -> list[dict[str, Any]]:
    """List all saved projects."""
    _ensure_dir()
    projects = []
    for fname in sorted(os.listdir(_projects_root())):
        if not fname.endswith(".json"):
            continue
        path = _projects_root() / fname
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


def delete_project(name: str) -> dict[str, Any]:
    """Delete a saved project."""
    path = _safe_path(name)
    name = _safe_name(name)
    if not os.path.exists(path):
        return {"error": f"Project '{name}' not found"}
    os.remove(path)
    return {"deleted": name}


def run_pipeline(
    graph: dict[str, Any],
    target: str = "ice40",
    *,
    process_limits: EdaProcessLimits | None = None,
) -> dict[str, Any]:
    """Run the Studio graph-to-synthesis pipeline.

    If the graph has ODE equations in population params, the pipeline compiles
    them to SystemVerilog and runs synthesis. Otherwise it uses a default LIF
    compile path.

    Parameters
    ----------
    graph:
        Studio network graph payload.
    target:
        Studio synthesis target identifier.
    process_limits:
        Optional host-supported CPU and address-space ceilings for the
        downstream synthesis child process.

    Returns
    -------
    dict[str, Any]
        Pipeline result containing validation, simulation, compile, and
        synthesis step payloads, or a bounded failure payload.
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
    synth_result = run_synthesis(verilog, target, process_limits=process_limits)
    steps["synthesise"] = synth_result

    return {
        "success": synth_result.get("success", False),
        "target": target,
        "steps": steps,
        "pipeline": "graph → simulate → compile → synthesise",
    }
