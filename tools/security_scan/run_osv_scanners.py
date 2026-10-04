#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — OSV scanner runner

"""Run OSV-Scanner and emit packet artefacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.security_scan.locked_dependency_inventory import inventory_locks

OSV_SCANNER_SCHEMA_VERSION = "sc-neurocore.osv-scanner.v1"
RunCommand = Callable[..., subprocess.CompletedProcess[str]]
SleepCommand = Callable[[float], None]
# Python profiles, Julia and Pixi have dedicated complete-audit adapters in
# locked_dependency_audit. This additional OSV lane uses supported extractors.


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the additional OSV security packet producer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_project_root(),
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Packet root; scanner artefacts are written under its security/ child.",
    )
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail_lines(text: str, *, limit: int = 12) -> list[str]:
    return text.strip().splitlines()[-limit:]


def _run(
    command: list[str],
    *,
    repo_root: Path,
    run_command: RunCommand,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _iter_packages(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    packages: list[dict[str, Any]] = []
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue
        for package in result.get("packages", []):
            if isinstance(package, dict):
                packages.append(package)
    return packages


def _vulnerability_ids(packages: list[dict[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for package in packages:
        for vulnerability in package.get("vulnerabilities", []):
            if not isinstance(vulnerability, dict):
                continue
            vuln_id = vulnerability.get("id")
            if isinstance(vuln_id, str) and vuln_id:
                ids.add(vuln_id)
    return sorted(ids)


def _validate_osv_report(path: Path) -> tuple[list[str], int, int, list[str]]:
    if not path.exists():
        return ["missing OSV report artifact"], 0, 0, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid JSON OSV report: {exc}"], 0, 0, []
    if not isinstance(payload, dict):
        return ["OSV report root must be an object"], 0, 0, []

    if not isinstance(payload.get("results"), list) or not payload["results"]:
        return ["OSV report has no audited package results"], 0, 0, []

    packages = _iter_packages(payload)
    if not packages:
        return ["OSV report has no audited packages"], 0, 0, []
    vulnerability_ids = _vulnerability_ids(packages)
    return [], len(packages), len(vulnerability_ids), vulnerability_ids


def _lockfile_arguments(repo_root: Path) -> list[str]:
    args: list[str] = []
    for lock in inventory_locks(repo_root):
        if lock.ecosystem in {"rust", "npm", "go"}:
            args.extend(("--lockfile", lock.path))
    return args


def _is_transient_osv_failure(result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode == 0:
        return False
    stderr = result.stderr.lower()
    return "rpc error" in stderr and (
        "service unavailable" in stderr
        or "code = unavailable" in stderr
        or "code = internal" in stderr
    )


def run_osv_scanner(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
    sleep: SleepCommand = time.sleep,
) -> dict[str, Any]:
    """Run all supported tracked locks and retain blocking scanner decisions.

    Parameters
    ----------
    repo_root : Path
        Checkout defining the tracked dependency inventory.
    output_dir : Path
        Security packet directory for the native JSON and summary.
    run_command : callable
        Bounded subprocess executor; defaults to the real scanner invocation.
    sleep : callable
        Backoff wait between retriable service failures.

    Returns
    -------
    dict
        Report custody and pass status. Nonzero tool exits always fail, even
        when partial or empty JSON contains no vulnerability records.
    """
    security_dir = output_dir / "security"
    security_dir.mkdir(parents=True, exist_ok=True)
    report_path = security_dir / "osv_scanner.json"
    command = [
        "osv-scanner",
        "scan",
        "source",
        "--config",
        "tools/security_scan/osv-scanner.toml",
        "--format",
        "json",
        "--all-packages",
        "--all-vulns",
        "--no-call-analysis",
        "go",
        "--output-file",
        str(report_path),
        "--experimental-no-default-plugins",
        "--experimental-plugins",
        "lockfile",
        *_lockfile_arguments(repo_root),
    ]
    attempts = 0
    result: subprocess.CompletedProcess[str] | None = None
    for attempts in range(1, 4):
        report_path.unlink(missing_ok=True)
        result = _run(command, repo_root=repo_root, run_command=run_command, timeout=360)
        if not _is_transient_osv_failure(result):
            break
        sleep(5 * attempts)
    assert result is not None
    validation_errors, package_count, vulnerability_count, vulnerability_ids = _validate_osv_report(
        report_path
    )
    clean_report = not validation_errors and vulnerability_count == 0
    process_ok = result.returncode == 0
    summary = {
        "schema_version": OSV_SCANNER_SCHEMA_VERSION,
        "passed": process_ok and clean_report,
        "command": command,
        "attempts": attempts,
        "artifact": str(report_path),
        "returncode": result.returncode,
        "stdout_tail": _tail_lines(result.stdout),
        "stderr_tail": _tail_lines(result.stderr),
        "package_count": package_count,
        "vulnerability_count": vulnerability_count,
        "vulnerability_ids": vulnerability_ids,
        "validation_errors": validation_errors,
    }
    _write_json(security_dir / "osv_scanner_summary.json", summary)
    return summary


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., dict[str, Any]] = run_osv_scanner,
) -> int:
    """Run the OSV packet CLI and propagate incomplete or vulnerable audits.

    Parameters
    ----------
    argv : list of str, optional
        Arguments for the repository and output directory.
    runner : callable
        Packet producer, using the real scanner by default.

    Returns
    -------
    int
        Zero only after the scanner succeeds and its report is clean.
    """
    args = build_parser().parse_args(argv)
    summary = runner(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
