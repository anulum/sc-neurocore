#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python code scanner plan generator

"""Build an offline deterministic scanner plan for Python code scanners."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, cast


SCAN_PLAN_SCHEMA_VERSION = "sc-neurocore.python-code-scanner-plan.v1"

TARGET_SCANNERS = (
    "bandit",
    "mypy",
    "pip-audit",
    "pyright",
    "ruff",
    "semgrep",
)

DEFERRED_HEAVY_SCANNERS: set[str] = set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline Python code scanner execution plan."
    )
    parser.add_argument("--output", type=Path, help="Write plan JSON to this path.")
    parser.add_argument(
        "--fail-on-missing-required-inputs",
        action="store_true",
        help=("Exit non-zero when any scanner is in missing_required_input state."),
    )
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_module_path() -> Path:
    return _project_root() / "tools" / "security_scanner_manifest.py"


def _load_manifest() -> dict[str, Any]:
    manifest_path = _manifest_module_path()
    spec = importlib.util.spec_from_file_location(
        "security_scanner_manifest_for_python_plan", manifest_path
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


def _is_tool_available(name: str) -> str | None:
    return shutil.which(name)


def _input_exists(path: Path) -> bool:
    return path.exists()


def _required_inputs(scanner: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for scanner_input in scanner.get("inputs", []):
        if not isinstance(scanner_input, dict):
            continue
        if scanner_input.get("required", True):
            path = scanner_input.get("path")
            if isinstance(path, str):
                result.append(path)
    return result


def _missing_required_inputs(scanner: dict[str, Any], repo_root: Path) -> list[str]:
    missing = [
        input_path
        for input_path in _required_inputs(scanner)
        if not _input_exists((repo_root / input_path).resolve())
    ]
    return sorted(missing)


def _manifest_entry_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for scanner in payload.get("scanners", []):
        if not isinstance(scanner, dict):
            continue
        name = scanner.get("name")
        if isinstance(name, str):
            by_name[name] = scanner
    return by_name


def _build_entry(
    scanner: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    name = str(scanner.get("name", ""))
    command = str(scanner.get("command", ""))
    executable = _extract_executable(command)

    missing_required_inputs = _missing_required_inputs(scanner, repo_root)
    if missing_required_inputs:
        return {
            "name": name,
            "run_class": "missing_required_input",
            "executable": executable,
            "missing_required_inputs": missing_required_inputs,
        }

    if name in DEFERRED_HEAVY_SCANNERS:
        return {
            "name": name,
            "run_class": "deferred_heavy",
            "executable": executable,
            "missing_required_inputs": [],
        }

    if executable and _is_tool_available(executable):
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


def build_scanner_plan(
    *,
    repo_root: Path,
    manifest_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest_payload if manifest_payload is not None else _load_manifest()
    manifest_by_name = _manifest_entry_by_name(manifest)

    scanners = sorted(
        [
            _build_entry(
                manifest_by_name.get(
                    name,
                    {
                        "name": name,
                        "command": name,
                        "inputs": [],
                    },
                ),
                repo_root=repo_root,
            )
            for name in TARGET_SCANNERS
        ],
        key=lambda entry: entry["name"],
    )

    return {
        "schema_version": SCAN_PLAN_SCHEMA_VERSION,
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

    plan = build_scanner_plan(repo_root=_project_root())
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
