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
SYFT_EXCLUDE_GLOBS = (
    "**/.git/**",
    "**/.cache/**",
    "**/.venv/**",
    "**/.venv-*/**",
    "**/.pixi/**",
    "**/build/**",
    "**/node_modules/**",
    "**/target/**",
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


def _project_identity(repo_root: Path) -> tuple[str, str]:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    return str(project["name"]), str(project["version"])


def _project_license(repo_root: Path) -> str:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(pyproject["project"]["license"])


def _component_licenses(component: dict[str, Any]) -> set[str]:
    licenses: set[str] = set()
    for entry in component.get("licenses", []):
        if not isinstance(entry, dict):
            continue
        license_value = entry.get("license")
        if not isinstance(license_value, dict):
            continue
        identifier = license_value.get("id") or license_value.get("name")
        if isinstance(identifier, str):
            licenses.add(identifier)
    return licenses


def _enrich_cyclonedx_root(
    path: Path,
    *,
    project_name: str,
    project_version: str,
    project_license: str,
) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []

    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return ["metadata must be an object for project enrichment"]
    component = metadata.setdefault(
        "component",
        {
            "type": "application",
            "name": project_name,
            "version": project_version,
        },
    )
    if not isinstance(component, dict):
        return ["metadata.component must be an object for project enrichment"]
    if project_license not in _component_licenses(component):
        licenses = component.setdefault("licenses", [])
        if not isinstance(licenses, list):
            return ["metadata.component.licenses must be a list"]
        licenses.append({"license": {"id": project_license}})
    _write_json(path, payload)
    return []


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


def _validate_cyclonedx_sbom(
    path: Path,
    *,
    project_name: str,
    project_version: str,
    project_license: str,
) -> tuple[list[str], int]:
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
    metadata = payload.get("metadata")
    root = metadata.get("component") if isinstance(metadata, dict) else None
    if not isinstance(root, dict):
        errors.append("metadata.component must be an object")
    else:
        if root.get("name") != project_name:
            errors.append(f"metadata.component.name must be {project_name!r}")
        if root.get("version") != project_version:
            errors.append(f"metadata.component.version must be {project_version!r}")
        if project_license not in _component_licenses(root):
            errors.append(f"metadata.component must include license {project_license!r}")
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
    project_license = _project_license(repo_root)
    command = [
        "syft",
        ".",
        "--source-name",
        project_name,
        "--source-version",
        project_version,
    ]
    for pattern in SYFT_EXCLUDE_GLOBS:
        command.extend(("--exclude", pattern))
    command.extend(("--output", f"cyclonedx-json={sbom_path}"))
    result = _run(command, repo_root=repo_root, run_command=run_command, timeout=300)
    enrichment_errors = _enrich_cyclonedx_root(
        sbom_path,
        project_name=project_name,
        project_version=project_version,
        project_license=project_license,
    )
    validation_errors, component_count = _validate_cyclonedx_sbom(
        sbom_path,
        project_name=project_name,
        project_version=project_version,
        project_license=project_license,
    )
    validation_errors = enrichment_errors + validation_errors
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
