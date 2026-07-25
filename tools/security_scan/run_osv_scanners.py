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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

OSV_SCANNER_SCHEMA_VERSION = "sc-neurocore.osv-scanner.v1"
RunCommand = Callable[..., subprocess.CompletedProcess[str]]
SleepCommand = Callable[[float], None]
# Only real lockfiles. The root requirements.txt is a short unpinned pointer
# (canonical pins live in pyproject / requirements/*.txt); scanning it as a
# lockfile triggers remote resolution RPCs that flake as "service unavailable"
# and fails the lane even when vulnerability_count is zero.
OSV_LOCKFILE_INPUTS = (
    "Cargo.lock",
    "fuzz/Cargo.lock",
    "crates/tinysc_riscv/Cargo.lock",
    "crates/evo_substrate_core/Cargo.lock",
    "crates/stochastic_doctor_core/Cargo.lock",
    "crates/autonomous_learning/Cargo.lock",
    "crates/core_engine/Cargo.lock",
    "crates/neuro_symbolic/Cargo.lock",
    "src/sc_neurocore/accel/rust/Cargo.lock",
    "studio/frontend/package-lock.json",
    "requirements/docs.txt",
    "requirements/runtime.txt",
)


def build_parser() -> argparse.ArgumentParser:
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

    packages = _iter_packages(payload)
    vulnerability_ids = _vulnerability_ids(packages)
    return [], len(packages), len(vulnerability_ids), vulnerability_ids


def _lockfile_arguments(repo_root: Path) -> list[str]:
    args: list[str] = []
    for relative_path in OSV_LOCKFILE_INPUTS:
        if (repo_root / relative_path).exists():
            args.extend(("--lockfile", relative_path))
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
        result = _run(command, repo_root=repo_root, run_command=run_command, timeout=360)
        if not _is_transient_osv_failure(result):
            break
        sleep(5 * attempts)
    assert result is not None
    validation_errors, package_count, vulnerability_count, vulnerability_ids = _validate_osv_report(
        report_path
    )
    # osv-scanner returns non-zero for both real findings and transient RPC
    # resolution errors. When the JSON report validates with zero vulns, treat
    # residual service-unavailable noise as non-blocking (already retried).
    clean_report = not validation_errors and vulnerability_count == 0
    process_ok = result.returncode == 0 or (clean_report and _is_transient_osv_failure(result))
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
    args = build_parser().parse_args(argv)
    summary = runner(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
