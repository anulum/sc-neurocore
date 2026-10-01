#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — typing scanner runner

"""Capture fresh Pyright and strict Mypy diagnostics from actual tool runs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

TYPING_SCANNER_SCHEMA_VERSION = "sc-neurocore.typing-scanners.v2"
TYPING_TOOL_VERSIONS = {"pyright": "1.1.414", "mypy": "2.3.1"}
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for repository-wide or explicitly scoped typing reports.

    Returns
    -------
    argparse.ArgumentParser
        Repository, output, scope and process-budget arguments.
    """
    parser = argparse.ArgumentParser(description="Run typing scanners.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True, help="Packet root.")
    parser.add_argument("--paths", nargs="+", help="Explicit files; recorded as a scoped scan.")
    parser.add_argument("--pyright-timeout", type=float, default=300)
    parser.add_argument("--mypy-timeout", type=float, default=600)
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    sibling = Path(sys.executable).parent / name
    return str(sibling) if sibling.is_file() else name


def _execute(
    command: list[str], repo_root: Path, run_command: RunCommand, timeout: float
) -> dict[str, object]:
    try:
        completed = run_command(
            command, cwd=repo_root, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "execution_error": type(exc).__name__,
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "execution_error": None,
    }


def _pyright_report(stdout: str, returncode: object) -> tuple[object, bool]:
    try:
        payload: object = json.loads(stdout)
    except json.JSONDecodeError:
        return {"raw_stdout": stdout}, False
    if not isinstance(payload, dict):
        return payload, False
    summary = payload.get("summary")
    diagnostics = payload.get("generalDiagnostics")
    if not isinstance(summary, dict) or not isinstance(diagnostics, list):
        return payload, False
    keys = ("filesAnalyzed", "errorCount", "warningCount", "informationCount")
    if any(type(summary.get(key)) is not int or summary[key] < 0 for key in keys):
        return payload, False
    if summary["filesAnalyzed"] == 0 or payload.get("version") != TYPING_TOOL_VERSIONS["pyright"]:
        return payload, False
    counts = {"error": 0, "warning": 0, "information": 0}
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            return payload, False
        severity = diagnostic.get("severity")
        if not isinstance(severity, str) or severity not in counts:
            return payload, False
        if not isinstance(diagnostic.get("message"), str):
            return payload, False
        counts[severity] += 1
    valid = all(counts[key] == summary[key + "Count"] for key in counts)
    valid = valid and returncode == (1 if counts["error"] else 0)
    return payload, valid


def _mypy_report(stdout: str, returncode: object) -> tuple[list[object], bool]:
    diagnostics: list[object] = []
    try:
        for line in stdout.splitlines():
            if line.strip():
                diagnostics.append(json.loads(line))
    except json.JSONDecodeError:
        return diagnostics, False
    errors = 0
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            return diagnostics, False
        if not all(isinstance(diagnostic.get(key), str) for key in ("file", "message")):
            return diagnostics, False
        if not all(type(diagnostic.get(key)) is int for key in ("line", "column")):
            return diagnostics, False
        if diagnostic.get("severity") not in ("error", "note", "warning"):
            return diagnostics, False
        errors += diagnostic["severity"] == "error"
    return diagnostics, returncode == (1 if errors else 0)


def validate_typing_output(scanner: str, stdout: str, returncode: object) -> tuple[object, bool]:
    """Parse stored tool output and check its diagnostics against the exit status.

    Parameters
    ----------
    scanner : str
        ``pyright`` or ``mypy``, selecting the pinned tool's output format.
    stdout : str
        Actual captured standard output; Mypy emits one JSON object per line.
    returncode : object
        Actual integer exit status, or ``None`` after execution failure.

    Returns
    -------
    tuple
        Parsed diagnostics and their structural/exit-status consistency.
        Valid diagnostics with errors remain a failed scan.

    Raises
    ------
    ValueError
        The scanner has no declared typing output format.
    """
    if scanner == "pyright":
        payload, valid = _pyright_report(stdout, returncode)
    elif scanner == "mypy":
        payload, valid = _mypy_report(stdout, returncode)
    else:
        raise ValueError("Unknown typing scanner output format.")
    return payload, valid and type(returncode) is int


def run_typing_scanners(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
    paths: tuple[str, ...] = (),
    pyright_timeout: float = 300,
    mypy_timeout: float = 600,
) -> dict[str, object]:
    """Run both pinned tools and replace reports with current-run evidence.

    Parameters
    ----------
    repo_root : Path
        Repository whose existing typing policies are used unchanged.
    output_dir : Path
        Packet root; owned reports are written under ``security``.
    run_command : callable
        Process executor, defaulting to the actual subprocess implementation.
    paths : tuple of str
        Explicit paths for a scoped run; empty retains each tool's original scope.
    pyright_timeout, mypy_timeout : float
        Positive finite process budgets, including version checks.

    Returns
    -------
    dict
        Current commands, versions, scope, execution errors and report validity.
        A scoped pass never certifies the full repository typing baseline.

    Raises
    ------
    ValueError
        A timeout is not positive and finite.
    OSError
        The output cannot be written; tool startup failures remain in the report.
    """
    if any(not math.isfinite(t) or t <= 0 for t in (pyright_timeout, mypy_timeout)):
        raise ValueError("Typing process budgets must be positive and finite.")
    security = output_dir / "security"
    security.mkdir(parents=True, exist_ok=True)
    commands = {
        "pyright": ["--project", "pyrightconfig.json", "--outputjson", *paths],
        "mypy": ["--strict", "--explicit-package-bases", "--output=json", *(paths or (".",))],
    }
    scanners: list[dict[str, object]] = []
    for name, timeout in (("pyright", pyright_timeout), ("mypy", mypy_timeout)):
        tool = _resolve_tool(name)
        version = _execute([tool, "--version"], repo_root, run_command, timeout)
        version_words = str(version["stdout"]).split()
        version_valid = version["returncode"] == 0 and version_words[:2] == [
            name,
            TYPING_TOOL_VERSIONS[name],
        ]
        result = _execute([tool, *commands[name]], repo_root, run_command, timeout)
        stdout = str(result["stdout"])
        payload, report_valid = validate_typing_output(name, stdout, result["returncode"])
        if name == "pyright":
            artifact = security / "pyright.json"
            _write_json(artifact, payload)
        else:
            artifact = security / "mypy"
            _write_json(
                artifact / "index.json",
                {"diagnostics": payload, "execution": result, "version": version},
            )
        for stream in ("stdout", "stderr"):
            (security / f"{name}.{stream}.log").write_text(str(result[stream]), encoding="utf-8")
        scanners.append(
            {
                "name": name,
                "artifact": str(artifact),
                **result,
                "version_check": version,
                "version_valid": version_valid,
                "report_valid": report_valid,
                "passed": result["returncode"] == 0 and version_valid and report_valid,
            }
        )
    failed = [str(scanner["name"]) for scanner in scanners if not scanner["passed"]]
    summary: dict[str, object] = {
        "schema_version": TYPING_SCANNER_SCHEMA_VERSION,
        "passed": not failed,
        "scope": {"kind": "explicit-paths" if paths else "repository", "paths": list(paths)},
        "failed_scanners": failed,
        "scanner_count": len(scanners),
        "scanners": scanners,
    }
    _write_json(security / "typing_scanner_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Execute the typing CLI and fail on diagnostics or invalid tool evidence.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments; omitted reads the process arguments.

    Returns
    -------
    int
        Zero only when both current tool runs pass for the recorded scope.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_typing_scanners(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            paths=tuple(args.paths or ()),
            pyright_timeout=args.pyright_timeout,
            mypy_timeout=args.mypy_timeout,
        )
    except ValueError:
        parser.error("Typing process budgets must be positive and finite.")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
