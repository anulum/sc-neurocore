#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust supply-chain scanner plan generator

"""Build an offline deterministic scanner plan for Rust and supply-chain gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

SCAN_PLAN_SCHEMA_VERSION = "sc-neurocore.security-supply-chain-plan.v1"

TARGET_SCANNERS = (
    "actionlint",
    "cargo-audit",
    "cargo-deny",
    "cargo-fuzz-nightly",
    "osv-scanner",
    "reuse",
    "trivy fs",
    "syft-cyclonedx",
)

HEAVY_SCANNERS = frozenset({"trivy fs", "cargo-fuzz-nightly"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline Rust supply-chain scanner execution plan."
    )
    parser.add_argument("--output", type=Path, help="Write plan JSON to this path.")
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Include heavy scanners in normal availability checks.",
    )
    parser.add_argument(
        "--fail-on-missing-required-inputs",
        action="store_true",
        help="Exit with non-zero status if any run_class is missing_required_input.",
    )
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_module_path() -> Path:
    return _project_root() / "tools" / "security_scanner_manifest.py"


def _load_manifest() -> dict[str, Any]:
    manifest_path = _manifest_module_path()
    spec = importlib.util.spec_from_file_location(
        "security_scanner_manifest_for_plan", manifest_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load security scanner manifest module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(dict[str, Any], module.build_scanner_manifest())


def _extract_executable(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return parts[0] if parts else ""


def _has_executable(name: str) -> str | None:
    return shutil.which(name)


def _required_inputs(scanner: dict[str, Any]) -> list[dict[str, Any]]:
    raw_inputs = scanner.get("inputs", [])
    if not isinstance(raw_inputs, list):
        return []

    required: list[dict[str, Any]] = []
    for scanner_input in raw_inputs:
        if not isinstance(scanner_input, dict):
            continue
        if scanner_input.get("required", True):
            required.append(scanner_input)
    return required


def _missing_required_inputs(scanner: dict[str, Any], repo_root: Path) -> list[str]:
    missing: list[str] = []
    for required in _required_inputs(scanner):
        path = required.get("path")
        if not isinstance(path, str):
            continue
        if not (repo_root / path).resolve().exists():
            missing.append(path)
    return sorted(missing)


def _build_entry(
    scanner: dict[str, Any],
    *,
    include_heavy: bool,
    repo_root: Path,
    has_executable: Callable[[str], str | None],
) -> dict[str, Any]:
    name = str(scanner.get("name", ""))
    command = str(scanner.get("command", ""))
    executable = _extract_executable(command)

    if name in HEAVY_SCANNERS and not include_heavy:
        return {
            "name": name,
            "run_class": "deferred_heavy",
            "executable": executable,
            "missing_required_inputs": [],
        }

    missing_required_inputs = _missing_required_inputs(scanner, repo_root)
    if missing_required_inputs:
        return {
            "name": name,
            "run_class": "missing_required_input",
            "executable": executable,
            "missing_required_inputs": missing_required_inputs,
        }

    if executable and has_executable(executable):
        return {
            "name": name,
            "run_class": "available",
            "executable": executable,
            "missing_required_inputs": [],
        }

    return {
        "name": name,
        "run_class": "missing_tool",
        "executable": executable,
        "missing_required_inputs": [],
    }


def _manifest_entry_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for entry in payload.get("scanners", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str):
            by_name[name] = entry
    return by_name


def build_rust_supply_chain_plan(
    manifest_payload: dict[str, Any],
    *,
    repo_root: Path,
    include_heavy: bool,
    has_executable: Callable[[str], str | None] = _has_executable,
) -> dict[str, Any]:
    scanner_lookup = _manifest_entry_by_name(manifest_payload)
    scanners = sorted(
        [
            _build_entry(
                scanner_lookup.get(name, {"name": name, "command": name}),
                include_heavy=include_heavy,
                repo_root=repo_root,
                has_executable=has_executable,
            )
            for name in TARGET_SCANNERS
        ],
        key=lambda entry: entry["name"],
    )
    return {
        "schema_version": SCAN_PLAN_SCHEMA_VERSION,
        "include_heavy": include_heavy,
        "scanner_count": len(scanners),
        "scanners": scanners,
    }


def _has_missing_required_inputs(plan: dict[str, Any]) -> bool:
    return any(
        scanner.get("run_class") == "missing_required_input"
        for scanner in plan.get("scanners", [])
        if isinstance(scanner, dict)
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest = _load_manifest()
    plan = build_rust_supply_chain_plan(
        manifest,
        repo_root=_project_root(),
        include_heavy=args.include_heavy,
        has_executable=_has_executable,
    )

    payload = json.dumps(plan, sort_keys=True, indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")

    if args.fail_on_missing_required_inputs and _has_missing_required_inputs(plan):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
