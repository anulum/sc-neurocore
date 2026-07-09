#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust security scanner runner

"""Run blocking Rust security scanners and emit release packet artefacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

RUST_SCANNER_SCHEMA_VERSION = "sc-neurocore.rust-security-scanners.v1"
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Rust security scanners.")
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
    content = stdout.strip()
    if not content and stderr.strip().startswith("{"):
        lines = []
        for line in stderr.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                lines.append(json.loads(stripped))
            except json.JSONDecodeError:
                return {"raw_stderr": stderr}
        return lines

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"raw_stdout": stdout}


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


def _record_result(
    *,
    name: str,
    artifact: Path,
    result: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    _write_json(artifact, _normalise_output(result.stdout, result.stderr))
    return {
        "name": name,
        "artifact": str(artifact),
        "returncode": result.returncode,
        "stderr": result.stderr,
    }


def run_rust_scanners(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    security_dir = output_dir / "security"
    security_dir.mkdir(parents=True, exist_ok=True)

    cargo_audit = _run(
        ["cargo", "audit", "--format", "json", "--file", "Cargo.lock"],
        repo_root=repo_root,
        run_command=run_command,
        timeout=180,
    )
    cargo_deny = _run(
        [
            "cargo",
            "deny",
            "--format",
            "json",
            "--manifest-path",
            "engine/Cargo.toml",
            "check",
            "--config",
            "engine/deny.toml",
            "licenses",
        ],
        repo_root=repo_root,
        run_command=run_command,
        timeout=240,
    )

    scanner_results = [
        _record_result(
            name="cargo-audit",
            artifact=security_dir / "cargo_audit.json",
            result=cargo_audit,
        ),
        _record_result(
            name="cargo-deny",
            artifact=security_dir / "cargo_deny.json",
            result=cargo_deny,
        ),
    ]

    failed = [scanner["name"] for scanner in scanner_results if scanner["returncode"] != 0]
    summary = {
        "schema_version": RUST_SCANNER_SCHEMA_VERSION,
        "passed": not failed,
        "failed_scanners": failed,
        "scanner_count": len(scanner_results),
        "scanners": scanner_results,
    }
    _write_json(security_dir / "rust_scanner_summary.json", summary)
    return summary


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., dict[str, Any]] = run_rust_scanners,
) -> int:
    args = build_parser().parse_args(argv)
    summary = runner(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
