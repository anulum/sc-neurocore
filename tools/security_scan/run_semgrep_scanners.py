#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Run the release Semgrep lane against repo-owned policy."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

SEMGREP_SCANNER_SCHEMA_VERSION = "sc-neurocore.semgrep-scanner.v1"
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _load_semgrep_findings(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    results = payload.get("results") if isinstance(payload, dict) else None
    return len(results) if isinstance(results, list) else 0


def _tail_lines(text: str, *, limit: int = 12) -> list[str]:
    return text.strip().splitlines()[-limit:]


def run_semgrep_scanner(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    """Run Semgrep with the repository-owned release policy."""
    security_dir = output_dir / "security"
    security_dir.mkdir(parents=True, exist_ok=True)

    semgrep_output = security_dir / "semgrep.json"
    policy_path = repo_root / ".semgrep.yml"
    command = [
        "semgrep",
        "scan",
        "--config",
        str(policy_path),
        "--json",
        "--error",
        "--output",
        str(semgrep_output),
        "src",
        "tools",
    ]
    result = _run(command, repo_root=repo_root, run_command=run_command, timeout=600)
    findings = _load_semgrep_findings(semgrep_output)
    summary = {
        "schema_version": SEMGREP_SCANNER_SCHEMA_VERSION,
        "passed": result.returncode == 0,
        "command": command,
        "artifact": str(semgrep_output),
        "finding_count": findings,
        "returncode": result.returncode,
        "stdout_tail": _tail_lines(result.stdout),
        "stderr_tail": _tail_lines(result.stderr),
    }
    _write_json(security_dir / "semgrep_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Semgrep scanner lane."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_project_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., dict[str, Any]] = run_semgrep_scanner,
) -> int:
    """Run the Semgrep scanner command."""
    args = build_parser().parse_args(argv)
    summary = runner(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
