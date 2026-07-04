#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Run the pinned Trivy filesystem vulnerability lane."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

TRIVY_FS_SCANNER_SCHEMA_VERSION = "sc-neurocore.trivy-fs-scanner.v1"
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _project_root() -> Path:
    """Return the repository root that owns this scanner lane."""
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON to a scanner artifact path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _text_output(value: object) -> str:
    """Return subprocess output as text even when timeout output is bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return ""


def _tail_lines(text: str, *, limit: int = 12) -> list[str]:
    """Return the final scanner output lines for compact release evidence."""
    return text.strip().splitlines()[-limit:]


def _run(
    command: list[str],
    *,
    repo_root: Path,
    run_command: RunCommand,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a scanner command and normalize timeout failures into a result."""
    try:
        return run_command(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=_text_output(exc.stdout),
            stderr="\n".join(
                part
                for part in (
                    _text_output(exc.stderr),
                    f"command timed out after {timeout} seconds",
                )
                if part
            ),
        )


def _iter_vulnerabilities(payload: Any) -> list[dict[str, Any]]:
    """Extract Trivy vulnerability objects from a report payload."""
    if not isinstance(payload, dict):
        return []
    vulnerabilities: list[dict[str, Any]] = []
    for result in payload.get("Results", []):
        if not isinstance(result, dict):
            continue
        for vulnerability in result.get("Vulnerabilities", []):
            if isinstance(vulnerability, dict):
                vulnerabilities.append(vulnerability)
    return vulnerabilities


def _vulnerability_ids(vulnerabilities: list[dict[str, Any]]) -> list[str]:
    """Return sorted vulnerability IDs from Trivy vulnerability objects."""
    ids: set[str] = set()
    for vulnerability in vulnerabilities:
        vulnerability_id = vulnerability.get("VulnerabilityID")
        if isinstance(vulnerability_id, str) and vulnerability_id:
            ids.add(vulnerability_id)
    return sorted(ids)


def _validate_trivy_report(path: Path) -> tuple[list[str], int, list[str]]:
    """Validate the Trivy JSON report and return errors plus vulnerability data."""
    if not path.exists():
        return ["missing Trivy filesystem report artifact"], 0, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid JSON Trivy filesystem report: {exc}"], 0, []
    if not isinstance(payload, dict):
        return ["Trivy filesystem report root must be an object"], 0, []
    vulnerabilities = _iter_vulnerabilities(payload)
    return [], len(vulnerabilities), _vulnerability_ids(vulnerabilities)


def run_trivy_fs_scanner(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    """Run Trivy filesystem scanning for high and critical fixed vulnerabilities."""
    security_dir = output_dir / "security"
    security_dir.mkdir(parents=True, exist_ok=True)
    report_path = security_dir / "trivy_fs.json"
    command = [
        "trivy",
        "fs",
        "--format",
        "json",
        "--output",
        str(report_path),
        "--exit-code",
        "1",
        "--severity",
        "HIGH,CRITICAL",
        "--ignore-unfixed",
        "--scanners",
        "vuln",
        str(repo_root),
    ]
    result = _run(command, repo_root=repo_root, run_command=run_command, timeout=900)
    validation_errors, vulnerability_count, vulnerability_ids = _validate_trivy_report(report_path)
    summary = {
        "schema_version": TRIVY_FS_SCANNER_SCHEMA_VERSION,
        "passed": (
            result.returncode == 0 and not validation_errors and vulnerability_count == 0
        ),
        "command": command,
        "artifact": str(report_path),
        "returncode": result.returncode,
        "stdout_tail": _tail_lines(result.stdout),
        "stderr_tail": _tail_lines(result.stderr),
        "vulnerability_count": vulnerability_count,
        "vulnerability_ids": vulnerability_ids,
        "validation_errors": validation_errors,
    }
    _write_json(security_dir / "trivy_fs_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Trivy filesystem scanner lane."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_project_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., dict[str, Any]] = run_trivy_fs_scanner,
) -> int:
    """Run the Trivy filesystem scanner command."""
    args = build_parser().parse_args(argv)
    summary = runner(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
