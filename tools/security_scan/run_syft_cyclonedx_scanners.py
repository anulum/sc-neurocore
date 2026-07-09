#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Syft CycloneDX scanner runner

"""Run Syft CycloneDX SBOM generation and validate the output contract."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:  # pragma: no cover - covered by Python-version matrix.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

SYFT_CYCLONEDX_SCHEMA_VERSION = "sc-neurocore.syft-cyclonedx-scanner.v1"
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


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


def _project_identity(repo_root: Path) -> tuple[str, str]:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    return str(project["name"]), str(project["version"])


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


def _validate_cyclonedx_sbom(path: Path) -> tuple[list[str], int]:
    if not path.exists():
        return ["missing SBOM artifact"], 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid JSON SBOM: {exc}"], 0
    if not isinstance(payload, dict):
        return ["SBOM root must be an object"], 0

    errors: list[str] = []
    if payload.get("bomFormat") != "CycloneDX":
        errors.append("bomFormat must be CycloneDX")
    spec_version = payload.get("specVersion")
    if not isinstance(spec_version, str) or not spec_version:
        errors.append("specVersion must be a non-empty string")
    components = payload.get("components", [])
    if components is None:
        components = []
    if not isinstance(components, list):
        errors.append("components must be a list")
        component_count = 0
    else:
        component_count = len(components)
    return errors, component_count


def run_syft_cyclonedx_scanner(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    security_dir = output_dir / "security"
    security_dir.mkdir(parents=True, exist_ok=True)
    sbom_path = security_dir / "sbom.cdx.json"
    project_name, project_version = _project_identity(repo_root)
    command = [
        "syft",
        ".",
        "--source-name",
        project_name,
        "--source-version",
        project_version,
        "--output",
        f"cyclonedx-json={sbom_path}",
    ]
    result = _run(command, repo_root=repo_root, run_command=run_command, timeout=300)
    validation_errors, component_count = _validate_cyclonedx_sbom(sbom_path)
    summary = {
        "schema_version": SYFT_CYCLONEDX_SCHEMA_VERSION,
        "passed": result.returncode == 0 and not validation_errors,
        "command": command,
        "artifact": str(sbom_path),
        "returncode": result.returncode,
        "stdout_tail": _tail_lines(result.stdout),
        "stderr_tail": _tail_lines(result.stderr),
        "component_count": component_count,
        "validation_errors": validation_errors,
    }
    _write_json(security_dir / "syft_cyclonedx_summary.json", summary)
    return summary


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., dict[str, Any]] = run_syft_cyclonedx_scanner,
) -> int:
    args = build_parser().parse_args(argv)
    summary = runner(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
