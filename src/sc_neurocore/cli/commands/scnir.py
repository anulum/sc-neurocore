# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NIR document command

"""Validate, convert, upgrade, and audit SC-NIR documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

_ACTIONS = (
    "validate",
    "upgrade",
    "export",
    "audit-hdl",
    "compatibility",
    "closure-audit",
)


def add_scnir_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register SC-NIR document operations.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "scnir",
        help="Validate, upgrade, export, or audit SC-NIR metadata",
        description="Operate on SC-aware NIR documents and HDL handoff evidence.",
    )
    parser.add_argument("action", nargs="?", choices=_ACTIONS, help="SC-NIR operation")
    parser.add_argument("scnir_path", nargs="?", help="Document, NIR model, or evidence root")
    parser.add_argument("--output", "-o", default=None, help="Optional output document or report")
    parser.add_argument("--dt", type=float, default=1.0, help="NIR simulation timestep")
    parser.add_argument("--T", type=int, default=256, help="Stochastic bitstream length")
    parser.set_defaults(handler=run_scnir)


def run_scnir(args: argparse.Namespace) -> int:
    """Execute an SC-NIR document operation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``scnir`` arguments.

    Returns
    -------
    int
        Zero on success, otherwise one for invalid input or evidence.
    """
    from sc_neurocore.ir import (
        SCNIRConversionConfig,
        SCNIRValidationError,
        build_scnir_compatibility_audit,
        export_scnir_from_nir,
        load_scnir,
        scnir_compatibility_matrix_dicts,
        upgrade_scnir_dict,
        validate_scnir_compatibility_matrix,
    )

    action = args.action
    path = args.scnir_path
    if action is None or (action not in {"compatibility", "closure-audit"} and path is None):
        _print_usage()
        return 1

    if action == "compatibility":
        evidence_root = Path(path) if path else Path.cwd()
        try:
            validate_scnir_compatibility_matrix(evidence_root=evidence_root)
            if args.output is not None:
                _write_json(Path(args.output), scnir_compatibility_matrix_dicts())
        except (OSError, ValueError, TypeError) as exc:
            print(f"SC-NIR compatibility matrix invalid: {exc}")
            return 1

        suffix = f"; report written: {args.output}" if args.output is not None else ""
        print(f"SC-NIR compatibility matrix valid: {evidence_root}{suffix}")
        return 0

    if action == "closure-audit":
        evidence_root = Path(path) if path else Path.cwd()
        try:
            closure_report = build_scnir_compatibility_audit(evidence_root=evidence_root)
            if args.output is not None:
                _write_json(Path(args.output), closure_report)
        except (OSError, ValueError, TypeError) as exc:
            print(f"SC-NIR closure audit invalid: {exc}")
            return 1

        suffix = f"; report written: {args.output}" if args.output is not None else ""
        print(
            "SC-NIR closure audit valid: "
            f"{evidence_root} ({closure_report['primitive_count']} primitive(s), "
            f"{closure_report['audit_evidence_file_count']} evidence file(s)){suffix}"
        )
        return 0

    path = cast(str, path)
    if action == "validate":
        try:
            document = load_scnir(path)
        except (OSError, SCNIRValidationError, ValueError) as exc:
            print(f"SC-NIR invalid: {exc}")
            return 1
        print(f"SC-NIR valid: {path} ({len(document.streams)} stream(s))")
        return 0

    if action == "upgrade":
        if args.output is None:
            print("Error: scnir upgrade requires --output upgraded.scnir.json")
            return 1
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("SC-NIR document must be a JSON object")
            payload = upgrade_scnir_dict(raw)
        except (OSError, SCNIRValidationError, ValueError, TypeError) as exc:
            print(f"SC-NIR upgrade failed: {exc}")
            return 1
        _write_json(Path(args.output), payload)
        print(f"SC-NIR upgraded: {args.output} ({len(payload['streams'])} stream(s))")
        return 0

    if action == "audit-hdl":
        from sc_neurocore.ir import SCNIRHDLHandoffAuditError, audit_scnir_hdl_handoff

        try:
            handoff_report = audit_scnir_hdl_handoff(path)
            if args.output is not None:
                _write_json(Path(args.output), handoff_report.as_dict())
        except (OSError, SCNIRHDLHandoffAuditError, ValueError, TypeError) as exc:
            print(f"SC-NIR HDL handoff invalid: {exc}")
            return 1
        suffix = f"; report written: {args.output}" if args.output is not None else ""
        print(
            "SC-NIR HDL handoff valid: "
            f"{path} ({handoff_report.stream_count} stream(s), "
            f"{handoff_report.source_module_count} source module(s)){suffix}"
        )
        return 0

    if args.output is None:
        print("Error: scnir export requires --output model.scnir.json")
        return 1
    try:
        document = export_scnir_from_nir(
            path,
            output_path=args.output,
            config=SCNIRConversionConfig(bitstream_length=int(args.T)),
            dt=float(args.dt),
        )
    except (OSError, SCNIRValidationError, ValueError, ImportError) as exc:
        print(f"SC-NIR export failed: {exc}")
        return 1
    print(f"SC-NIR exported: {args.output} ({len(document.streams)} stream(s))")
    return 0


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_usage() -> None:
    print("Error: usage: sc-neurocore scnir validate model.scnir.json")
    print("       or: sc-neurocore scnir upgrade model.scnir.json --output upgraded.scnir.json")
    print("       or: sc-neurocore scnir export model.nir --output model.scnir.json")
    print("       or: sc-neurocore scnir audit-hdl build/ --output scnir_audit.json")
    print("       or: sc-neurocore scnir compatibility [repo-root]")
    print("       or: sc-neurocore scnir closure-audit [repo-root] --output scnir_audit.json")
