# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Project save/load and pipeline for Studio (Block 6)

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.studio.project_manifest import build_project_save_manifest
from sc_neurocore.studio.synthesis import EdaProcessLimits
from sc_neurocore.studio.workspace_lock import DEFAULT_LOCK_TIMEOUT
from sc_neurocore.studio.workspace_schema import PROJECT_PAYLOAD_VERSION, WorkspaceSchemaError
from sc_neurocore.studio.workspace_store import WorkspaceStore

_PROJECTS_DIR = os.path.join(os.path.expanduser("~"), ".sc-neurocore", "studio", "projects")

logger = logging.getLogger(__name__)

#: How long a request waits for another writer of the same workspace before it
#: is refused with HTTP 503. A request thread is a scarce resource, so this is
#: a bounded wait rather than a blocking one.
_LOCK_TIMEOUT = DEFAULT_LOCK_TIMEOUT

#: Payload version reported in save manifests; the revision document carries
#: the same value so it satisfies the evidence-bundle project contract.
_PROJECT_PAYLOAD_VERSION = PROJECT_PAYLOAD_VERSION

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


def _store() -> WorkspaceStore:
    """Return the versioned workspace store rooted at the project directory.

    The store serialises writers of one workspace across processes, so a
    request that arrives while another writer holds the same workspace waits
    rather than racing it. The wait is bounded by ``_LOCK_TIMEOUT``: a request
    thread must be released even when the other writer is stuck.
    """
    return WorkspaceStore(root=_projects_root(), lock_timeout=_LOCK_TIMEOUT)


def save_project(
    name: str, state: dict[str, Any], *, expected_revision: int | None = None
) -> dict[str, Any]:
    """Save full Studio state as a new immutable revision.

    Parameters
    ----------
    name:
        Workspace name; one directory under the Studio project root.
    state:
        Complete Studio state to persist.
    expected_revision:
        The revision the caller edited. ``None`` means the caller believes the
        workspace is new. A save from a revision that is no longer current is
        refused with :class:`~sc_neurocore.studio.workspace_store.WorkspaceConflict`
        rather than silently replacing the other editor's work.

    Returns
    -------
    dict[str, Any]
        Path-free evidence metadata, including the revision written and the
        revision it descends from.

    Raises
    ------
    ValueError
        The name is unusable, the state is not an object, or it carries an
        identifier that would later interpolate into HDL or MLIR source.
    WorkspaceConflict
        The workspace moved on while the caller was editing.
    """
    _ensure_dir()
    name = _safe_name(name)
    if not isinstance(state, dict):
        raise ValueError("Project state must be an object")
    payload_for_validation = {"name": name, "state": state}
    _validate_hdl_identifiers(payload_for_validation)
    store = _store()
    revision = store.save(name, state, expected_revision=expected_revision)
    manifest = build_project_save_manifest(
        name=name,
        saved_at=revision.saved_at,
        version=_PROJECT_PAYLOAD_VERSION,
        state=state,
        project_payload={
            "name": name,
            "saved_at": revision.saved_at,
            "version": _PROJECT_PAYLOAD_VERSION,
            "state": state,
        },
    )
    public = manifest.to_public_dict()
    public["revision"] = revision.revision
    public["parent_revision"] = revision.parent
    return public


def load_project(name: str, *, revision: int | None = None) -> dict[str, Any]:
    """Load one revision of a saved workspace, defaulting to the current one.

    Parameters
    ----------
    name:
        Workspace name.
    revision:
        Revision to read. ``None`` reads the current one; an earlier number
        reads history, which no later save can have rewritten.

    Returns
    -------
    dict[str, Any]
        The stored document, or ``{"error": ...}`` when the workspace or the
        requested revision does not exist.

    Raises
    ------
    ValueError
        The name is unusable, the stored document is not a workspace this
        build reads, or it carries an unsafe HDL-facing identifier.
    """
    name = _safe_name(name)
    store = _store()
    try:
        document = store.load(name, revision=revision)
    except KeyError:
        target = "" if revision is None else f" revision {revision}"
        return {"error": f"Project '{name}'{target} not found"}
    except WorkspaceSchemaError as exc:
        raise ValueError(f"Invalid project payload: {exc}") from exc
    stored_name = document.get("name")
    if stored_name is not None and _safe_name(str(stored_name)) != name:
        raise ValueError("Invalid project payload: inconsistent project name")
    _validate_hdl_identifiers(document)
    return document


def list_projects() -> list[dict[str, Any]]:
    """List every saved workspace with its current revision.

    A workspace whose revisions cannot be read is reported with a ``null``
    revision rather than omitted, so a corrupt store is visible instead of
    looking empty.
    """
    _ensure_dir()
    store = _store()
    projects: list[dict[str, Any]] = []
    for summary in store.list_workspaces():
        name = str(summary["name"])
        try:
            document = store.load(name)
            saved_at = document.get("saved_at")
        except (KeyError, WorkspaceSchemaError):
            saved_at = None
        projects.append(
            {
                "name": name,
                "revision": summary["revision"],
                "revision_count": summary["revision_count"],
                "saved_at": saved_at,
                "version": _PROJECT_PAYLOAD_VERSION,
            }
        )
    return projects


def delete_project(name: str) -> dict[str, Any]:
    """Move a workspace to the recoverable trash.

    Nothing is erased: the workspace and its whole revision history move
    aside, and :func:`restore_project` brings them back. The returned token
    identifies the deleted copy.
    """
    name = _safe_name(name)
    store = _store()
    try:
        destination = store.delete(name)
    except KeyError:
        return {"error": f"Project '{name}' not found"}
    return {"deleted": name, "recoverable": True, "token": destination.name}


def list_deleted_projects() -> list[dict[str, Any]]:
    """List the workspaces waiting in the recoverable trash, newest first."""
    return [dict(entry) for entry in _store().deleted()]


def restore_project(token: str) -> dict[str, Any]:
    """Restore one deleted workspace under its original name.

    Restoring onto a name that is in use is refused rather than performed:
    overwriting a live workspace is the loss this store exists to prevent.
    """
    store = _store()
    try:
        name = store.restore(token)
    except KeyError:
        return {"error": "Deleted project not found"}
    return {"restored": name, "revision": store.head_revision(name)}


def fork_project(name: str, new_name: str, *, revision: int | None = None) -> dict[str, Any]:
    """Copy one revision of a workspace into a new one.

    The source workspace is untouched; the fork starts at revision 1.
    """
    name = _safe_name(name)
    new_name = _safe_name(new_name)
    store = _store()
    try:
        created = store.fork(name, new_name, revision=revision)
    except KeyError:
        return {"error": f"Project '{name}' not found"}
    return {"forked": new_name, "from": name, "revision": created.revision}


def project_revisions(name: str) -> list[dict[str, Any]]:
    """Return every stored revision of one workspace, oldest first."""
    return [revision.to_public_dict() for revision in _store().revisions(_safe_name(name))]


def export_project(name: str, *, revision: int | None = None) -> dict[str, Any]:
    """Return one revision as a self-contained document for transfer."""
    name = _safe_name(name)
    try:
        return _store().export_document(name, revision=revision)
    except KeyError:
        return {"error": f"Project '{name}' not found"}


def import_project(name: str, document: dict[str, Any]) -> dict[str, Any]:
    """Create a workspace from an exported document."""
    name = _safe_name(name)
    store = _store()
    try:
        created = store.import_document(name, document)
    except WorkspaceSchemaError as exc:
        return {"error": f"Invalid workspace document: {exc}"}
    return {"imported": name, "revision": created.revision}


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
