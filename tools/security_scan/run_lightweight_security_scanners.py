#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Run lightweight blocking security scanners and emit release packet artefacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

LIGHTWEIGHT_SCANNER_SCHEMA_VERSION = "sc-neurocore.lightweight-security-scanners.v1"
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lightweight security scanners and write JSON artefacts."
    )
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


def _parse_actionlint_json_lines(stdout: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = {"message": stripped}
        if isinstance(payload, dict):
            findings.append(payload)
    return findings


def run_lightweight_scanners(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    security_dir = output_dir / "security"
    security_dir.mkdir(parents=True, exist_ok=True)

    scanner_results: list[dict[str, Any]] = []

    ruff_output = security_dir / "ruff.json"
    ruff = _run(
        [
            "ruff",
            "check",
            "--output-format",
            "json",
            "--output-file",
            str(ruff_output),
            "--cache-dir",
            str(security_dir / "ruff-cache"),
            "src",
            "tools",
            "tests",
        ],
        repo_root=repo_root,
        run_command=run_command,
        timeout=120,
    )
    scanner_results.append(
        {
            "name": "ruff",
            "artifact": str(ruff_output),
            "returncode": ruff.returncode,
            "stderr": ruff.stderr,
        }
    )

    bandit_output = security_dir / "bandit.json"
    bandit = _run(
        [
            "bandit",
            "-q",
            "-r",
            "src/sc_neurocore",
            "tools",
            "-x",
            "src/sc_neurocore/accel/mojo/.pixi",
            "--severity-level",
            "medium",
            "--format",
            "json",
            "--output",
            str(bandit_output),
        ],
        repo_root=repo_root,
        run_command=run_command,
        timeout=300,
    )
    scanner_results.append(
        {
            "name": "bandit",
            "artifact": str(bandit_output),
            "returncode": bandit.returncode,
            "stderr": bandit.stderr,
        }
    )

    actionlint_output = security_dir / "actionlint.json"
    actionlint = _run(
        [
            "actionlint",
            "-shellcheck",
            "",
            "-pyflakes",
            "",
            "-format",
            "{{json .}}",
        ],
        repo_root=repo_root,
        run_command=run_command,
        timeout=120,
    )
    _write_json(actionlint_output, _parse_actionlint_json_lines(actionlint.stdout))
    scanner_results.append(
        {
            "name": "actionlint",
            "artifact": str(actionlint_output),
            "returncode": actionlint.returncode,
            "stderr": actionlint.stderr,
        }
    )

    failed = [scanner["name"] for scanner in scanner_results if scanner["returncode"] != 0]
    summary = {
        "schema_version": LIGHTWEIGHT_SCANNER_SCHEMA_VERSION,
        "passed": not failed,
        "failed_scanners": failed,
        "scanner_count": len(scanner_results),
        "scanners": scanner_results,
    }
    _write_json(security_dir / "lightweight_scanner_summary.json", summary)
    return summary


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., dict[str, Any]] = run_lightweight_scanners,
) -> int:
    args = build_parser().parse_args(argv)
    summary = runner(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
