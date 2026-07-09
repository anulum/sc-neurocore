#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python compliance scanner runner

"""Run Python dependency/compliance scanners and emit packet artefacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PYTHON_COMPLIANCE_SCHEMA_VERSION = "sc-neurocore.python-compliance-scanners.v1"
NON_BLOCKING_SCANNERS = frozenset({"reuse"})
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Python compliance scanners.")
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


def _normalise_output(stdout: str, stderr: str) -> Any:
    content = stdout.strip() or stderr.strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_stdout": stdout, "raw_stderr": stderr}


def _resolve_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    sibling = Path(sys.executable).resolve().parent / name
    if sibling.exists():
        return str(sibling)
    return name


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


def run_python_compliance_scanners(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    security_dir = output_dir / "security"
    security_dir.mkdir(parents=True, exist_ok=True)

    pip_audit_output = security_dir / "pip_audit.json"
    pip_audit = _run(
        [
            _resolve_tool("pip-audit"),
            "--strict",
            "--requirement",
            "requirements/release.txt",
            "--format",
            "json",
            "--progress-spinner",
            "off",
            "--output",
            str(pip_audit_output),
        ],
        repo_root=repo_root,
        run_command=run_command,
        timeout=240,
    )

    reuse_output = security_dir / "reuse.json"
    reuse = _run(
        [_resolve_tool("reuse"), "--root", ".", "lint", "--json"],
        repo_root=repo_root,
        run_command=run_command,
        timeout=180,
    )
    _write_json(reuse_output, _normalise_output(reuse.stdout, reuse.stderr))

    scanner_results = [
        {
            "name": "pip-audit",
            "artifact": str(pip_audit_output),
            "returncode": pip_audit.returncode,
            "stderr": pip_audit.stderr,
        },
        {
            "name": "reuse",
            "artifact": str(reuse_output),
            "returncode": reuse.returncode,
            "stderr": reuse.stderr,
        },
    ]
    all_failed = [
        scanner["name"]
        for scanner in scanner_results
        if scanner["returncode"] != 0 or not Path(str(scanner["artifact"])).exists()
    ]
    failed = [name for name in all_failed if name not in NON_BLOCKING_SCANNERS]
    non_blocking_failed = [name for name in all_failed if name in NON_BLOCKING_SCANNERS]
    summary = {
        "schema_version": PYTHON_COMPLIANCE_SCHEMA_VERSION,
        "passed": not failed,
        "failed_scanners": failed,
        "non_blocking_failed_scanners": non_blocking_failed,
        "scanner_count": len(scanner_results),
        "scanners": scanner_results,
    }
    _write_json(security_dir / "python_compliance_summary.json", summary)
    return summary


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., dict[str, Any]] = run_python_compliance_scanners,
) -> int:
    args = build_parser().parse_args(argv)
    summary = runner(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
