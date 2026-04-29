# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Offline supply-chain audit helper

"""Audit committed SBOM and release dependency artefacts without network access."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


_REQ_RE = re.compile(r"^([A-Za-z0-9_.-]+(?:\[[^\]]+\])?)==([^\s\\]+)")


@dataclass(frozen=True)
class Finding:
    """One supply-chain audit finding."""

    level: str
    message: str


def audit_supply_chain(
    *,
    sbom_path: Path,
    pyproject_path: Path,
    requirements_path: Path,
    strict: bool = False,
) -> dict[str, Any]:
    """Return an offline supply-chain audit report."""
    findings: list[Finding] = []
    project = _load_pyproject(pyproject_path, findings)
    sbom = _load_json_object(sbom_path, "SBOM", findings)

    _audit_sbom(sbom, project, findings)
    _audit_release_requirements(requirements_path, findings)

    errors = sum(1 for finding in findings if finding.level == "error")
    warnings = sum(1 for finding in findings if finding.level == "warning")
    passed = errors == 0 and (warnings == 0 or not strict)
    return {
        "passed": passed,
        "strict": strict,
        "errors": errors,
        "warnings": warnings,
        "findings": [finding.__dict__ for finding in findings],
    }


def _load_pyproject(path: Path, findings: list[Finding]) -> dict[str, Any]:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        findings.append(Finding("error", f"{path}: cannot read project metadata: {exc}"))
        return {}
    project = payload.get("project")
    if not isinstance(project, dict):
        findings.append(Finding("error", f"{path}: missing [project] table"))
        return {}
    return project


def _load_json_object(path: Path, label: str, findings: list[Finding]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        findings.append(Finding("error", f"{path}: cannot read {label}: {exc}"))
        return {}
    if not isinstance(payload, dict):
        findings.append(Finding("error", f"{path}: {label} must be a JSON object"))
        return {}
    return payload


def _audit_sbom(sbom: dict[str, Any], project: dict[str, Any], findings: list[Finding]) -> None:
    if not sbom:
        return
    if sbom.get("bomFormat") != "CycloneDX":
        findings.append(Finding("error", "SBOM bomFormat must be CycloneDX"))
    if not str(sbom.get("specVersion", "")).startswith("1."):
        findings.append(Finding("error", "SBOM specVersion must be CycloneDX 1.x"))

    metadata = sbom.get("metadata")
    root = metadata.get("component") if isinstance(metadata, dict) else None
    if not isinstance(root, dict):
        findings.append(Finding("error", "SBOM metadata.component is missing"))
        root = {}

    project_name = project.get("name")
    if project_name and root.get("name") != project_name:
        findings.append(
            Finding("error", f"SBOM root component name {root.get('name')!r} != {project_name!r}")
        )

    project_version = project.get("version")
    if project_version and root.get("version") != project_version:
        findings.append(
            Finding(
                "warning",
                f"SBOM root component version {root.get('version')!r} != {project_version!r}",
            )
        )

    project_license = project.get("license")
    if isinstance(project_license, str) and project_license not in _component_licences(root):
        findings.append(
            Finding("warning", f"SBOM root component omits project licence {project_license!r}")
        )

    components = sbom.get("components")
    if not isinstance(components, list) or not components:
        findings.append(Finding("error", "SBOM components must be a non-empty list"))
        return

    seen_refs: set[str] = set()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            findings.append(Finding("error", f"SBOM component {index} must be an object"))
            continue
        name = component.get("name")
        if not isinstance(name, str) or not name:
            findings.append(Finding("error", f"SBOM component {index} has no name"))
        bom_ref = component.get("bom-ref")
        if isinstance(bom_ref, str) and bom_ref:
            if bom_ref in seen_refs:
                findings.append(Finding("error", f"duplicate SBOM component bom-ref {bom_ref!r}"))
            seen_refs.add(bom_ref)


def _component_licences(component: dict[str, Any]) -> set[str]:
    licences: set[str] = set()
    for entry in component.get("licenses", []):
        if not isinstance(entry, dict):
            continue
        licence = entry.get("license")
        if isinstance(licence, dict):
            value = licence.get("id") or licence.get("name")
            if isinstance(value, str):
                licences.add(value)
    return licences


def _audit_release_requirements(path: Path, findings: list[Finding]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        findings.append(Finding("error", f"{path}: cannot read release requirements: {exc}"))
        return
    if "pip-compile --generate-hashes" not in "\n".join(lines[:8]):
        findings.append(Finding("warning", f"{path}: header does not show --generate-hashes"))

    package_blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        match = _REQ_RE.match(line)
        if match:
            current = match.group(1).split("[", 1)[0].lower()
            package_blocks[current] = [line]
        elif current is not None:
            package_blocks[current].append(line)

    if not package_blocks:
        findings.append(Finding("error", f"{path}: no pinned packages found"))
        return

    for package, block in sorted(package_blocks.items()):
        joined = "\n".join(block)
        if "--hash=sha256:" not in joined:
            findings.append(Finding("error", f"{path}: {package} is pinned without sha256 hashes"))

    if "cyclonedx-bom" not in package_blocks:
        findings.append(Finding("warning", f"{path}: cyclonedx-bom is not pinned"))


def _print_human(report: dict[str, Any]) -> None:
    state = "PASS" if report["passed"] else "FAIL"
    print(f"supply-chain audit: {state} ({report['errors']} errors, {report['warnings']} warnings)")
    for finding in report["findings"]:
        print(f"- {finding['level']}: {finding['message']}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbom", type=Path, default=Path("sbom.cdx.json"))
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--requirements", type=Path, default=Path("requirements/release.txt"))
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the audit command."""
    args = build_parser().parse_args(argv)
    report = audit_supply_chain(
        sbom_path=args.sbom,
        pyproject_path=args.pyproject,
        requirements_path=args.requirements,
        strict=args.strict,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
