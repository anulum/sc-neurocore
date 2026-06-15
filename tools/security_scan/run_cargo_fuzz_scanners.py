#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Run bounded cargo-fuzz targets and emit release packet artefacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:  # pragma: no cover - covered by Python-version matrix.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

CARGO_FUZZ_SCHEMA_VERSION = "sc-neurocore.cargo-fuzz-scanners.v1"
CARGO_FUZZ_PROCESS_TIMEOUT_OVERHEAD_SECONDS = 300
CARGO_FUZZ_BUILD_TIMEOUT_SECONDS = 1800
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded cargo-fuzz targets.")
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
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Fuzz target to run; use 'all' to run every target from fuzz/Cargo.toml.",
    )
    parser.add_argument(
        "--max-total-time",
        type=int,
        default=300,
        help="Total seconds budget split across selected targets.",
    )
    parser.add_argument(
        "--build-timeout",
        type=int,
        default=CARGO_FUZZ_BUILD_TIMEOUT_SECONDS,
        help="Seconds allowed for each cargo-fuzz target build before execution.",
    )
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail_lines(text: str, *, limit: int = 12) -> list[str]:
    return text.strip().splitlines()[-limit:]


def discover_fuzz_targets(repo_root: Path) -> list[str]:
    manifest_path = repo_root / "fuzz" / "Cargo.toml"
    payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    bins = payload.get("bin", [])
    if not isinstance(bins, list):
        return []
    targets: list[str] = []
    for entry in bins:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        path = entry.get("path")
        if not isinstance(name, str) or not isinstance(path, str):
            continue
        if (repo_root / "fuzz" / path).exists():
            targets.append(name)
    return sorted(targets)


def _selected_targets(repo_root: Path, requested_targets: Sequence[str]) -> list[str]:
    available = discover_fuzz_targets(repo_root)
    if not requested_targets or "all" in requested_targets:
        return available

    unknown = sorted(set(requested_targets) - set(available))
    if unknown:
        raise ValueError(f"unknown fuzz targets: {', '.join(unknown)}")
    return sorted(set(requested_targets))


def _run(
    command: list[str],
    *,
    repo_root: Path,
    run_command: RunCommand,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
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
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr
        )
        timeout_message = f"command timed out after {timeout} seconds"
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout or "",
            stderr="\n".join(part for part in (stderr, timeout_message) if part),
        )


def _target_command(target: str, seconds: int) -> list[str]:
    return [
        "cargo",
        "+nightly",
        "fuzz",
        "run",
        target,
        "--fuzz-dir",
        "fuzz",
        "--",
        f"-max_total_time={seconds}",
    ]


def _target_build_command(target: str) -> list[str]:
    return [
        "cargo",
        "+nightly",
        "fuzz",
        "build",
        target,
        "--fuzz-dir",
        "fuzz",
    ]


def run_cargo_fuzz_scanners(
    *,
    repo_root: Path,
    output_dir: Path,
    selected_targets: Sequence[str] = ("all",),
    max_total_time: int = 300,
    build_timeout: int = CARGO_FUZZ_BUILD_TIMEOUT_SECONDS,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    targets = _selected_targets(repo_root, selected_targets)
    if not targets:
        raise ValueError("no fuzz targets discovered")
    if max_total_time < len(targets):
        raise ValueError("max_total_time must be at least the selected target count")
    if build_timeout < 1:
        raise ValueError("build_timeout must be positive")

    security_dir = output_dir / "security"
    security_dir.mkdir(parents=True, exist_ok=True)

    seconds_per_target = max(1, max_total_time // len(targets))
    target_results: list[dict[str, Any]] = []
    for target in targets:
        build_command = _target_build_command(target)
        build_result = _run(
            build_command,
            repo_root=repo_root,
            run_command=run_command,
            timeout=build_timeout,
        )
        command = _target_command(target, seconds_per_target)
        if build_result.returncode != 0:
            target_report = {
                "target": target,
                "phase": "build",
                "build_command": build_command,
                "build_returncode": build_result.returncode,
                "build_stdout_tail": _tail_lines(build_result.stdout),
                "build_stderr_tail": _tail_lines(build_result.stderr),
                "command": command,
                "returncode": build_result.returncode,
                "stdout_tail": [],
                "stderr_tail": _tail_lines(build_result.stderr),
                "passed": False,
            }
            _write_json(security_dir / f"cargo_fuzz_{target}.json", target_report)
            target_results.append(target_report)
            continue

        result = _run(
            command,
            repo_root=repo_root,
            run_command=run_command,
            timeout=seconds_per_target + CARGO_FUZZ_PROCESS_TIMEOUT_OVERHEAD_SECONDS,
        )
        target_report = {
            "target": target,
            "phase": "run",
            "build_command": build_command,
            "build_returncode": build_result.returncode,
            "build_stdout_tail": _tail_lines(build_result.stdout),
            "build_stderr_tail": _tail_lines(build_result.stderr),
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": _tail_lines(result.stdout),
            "stderr_tail": _tail_lines(result.stderr),
            "passed": result.returncode == 0,
        }
        _write_json(security_dir / f"cargo_fuzz_{target}.json", target_report)
        target_results.append(target_report)

    failed = [result["target"] for result in target_results if result["returncode"] != 0]
    summary = {
        "schema_version": CARGO_FUZZ_SCHEMA_VERSION,
        "passed": not failed,
        "failed_targets": failed,
        "target_count": len(target_results),
        "targets": target_results,
    }
    _write_json(security_dir / "cargo_fuzz_summary.json", summary)
    return summary


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., dict[str, Any]] = run_cargo_fuzz_scanners,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = runner(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            selected_targets=tuple(args.target or ("all",)),
            max_total_time=args.max_total_time,
            build_timeout=args.build_timeout,
        )
    except ValueError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
