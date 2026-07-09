#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — release security artifact index builder

"""Build and validate a deterministic offline release security artifact index."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION = "sc-neurocore.release-security-artifact-index.v1"


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Build a deterministic release security artifact index."
    )


def _resolve_artifact_root_path(root: Path, artifact_path: str) -> Path:
    candidate = Path(artifact_path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def validate_artifact_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    if payload.get("schema_version") != RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION:
        findings.append(
            {
                "level": "error",
                "message": "schema_version is missing or not equal to expected version",
            }
        )

    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, list):
        findings.append({"level": "error", "message": "artifacts must be a list"})
        return _final_report(findings)

    _validate_entries(
        payload=payload,
        key="artifacts",
        required_keys={"id", "path", "required"},
        findings=findings,
    )
    _validate_entries(
        payload=payload,
        key="vulnerability_status",
        required_keys={"id", "path", "required", "scanner"},
        findings=findings,
        optional=True,
    )

    return _final_report(findings)


def _validate_entries(
    *,
    payload: dict[str, Any],
    key: str,
    required_keys: set[str],
    findings: list[dict[str, str]],
    optional: bool = False,
) -> None:
    entries_raw = payload.get(key)
    singular = key[:-1] if key.endswith("s") else key
    if entries_raw is None and optional:
        return
    if not isinstance(entries_raw, list):
        findings.append({"level": "error", "message": f"{key} must be a list"})
        return

    entry_ids: set[str] = set()
    for entry in entries_raw:
        if not isinstance(entry, dict):
            findings.append({"level": "error", "message": f"{singular} entry must be an object"})
            continue

        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            findings.append(
                {"level": "error", "message": f"{singular}.id must be a non-empty string"}
            )
            continue

        missing = sorted(required_keys - set(entry))
        if missing:
            findings.append(
                {
                    "level": "error",
                    "message": f"{singular} {entry_id} missing fields: {', '.join(missing)}",
                }
            )

        entry_path = entry.get("path")
        if not isinstance(entry_path, str) or not entry_path:
            findings.append(
                {
                    "level": "error",
                    "message": f"{singular} {entry_id} path must be a non-empty string",
                }
            )

        if not isinstance(entry.get("required"), bool):
            findings.append(
                {
                    "level": "error",
                    "message": f"{singular} {entry_id} required must be true/false",
                }
            )

        scanner = entry.get("scanner")
        if "scanner" in required_keys and (not isinstance(scanner, str) or not scanner):
            findings.append(
                {
                    "level": "error",
                    "message": f"{singular} {entry_id} scanner must be a non-empty string",
                }
            )

        if entry_id in entry_ids:
            findings.append(
                {
                    "level": "error",
                    "message": f"duplicate {singular} id {entry_id}",
                }
            )
        entry_ids.add(entry_id)


def _final_report(findings: list[dict[str, str]]) -> dict[str, Any]:
    errors = [finding for finding in findings if finding["level"] == "error"]
    return {
        "schema_version": RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "passed": len(errors) == 0,
        "findings": findings,
        "errors": len(errors),
        "warnings": len([finding for finding in findings if finding["level"] == "warning"]),
    }


def build_artifact_index(
    manifest_payload: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    if not isinstance(root, Path):
        root = Path(root)

    report = validate_artifact_manifest(manifest_payload)
    if not report["passed"]:
        raise ValueError("invalid release artifact manifest")

    artifacts = manifest_payload["artifacts"]
    if not isinstance(artifacts, list):
        raise ValueError("invalid release artifact manifest")
    vulnerability_status_entries = manifest_payload.get("vulnerability_status", [])
    if not isinstance(vulnerability_status_entries, list):
        raise ValueError("invalid release artifact manifest")

    def _artifact_entry(raw_artifact: dict[str, Any]) -> dict[str, Any]:
        artifact_id = str(raw_artifact["id"])
        artifact_path = str(raw_artifact["path"])
        required = bool(raw_artifact["required"])
        normalized_path = Path(artifact_path)
        present = _resolve_artifact_root_path(root, artifact_path).exists()

        return {
            "id": artifact_id,
            "path": str(normalized_path),
            "required": required,
            "present": present,
        }

    built_artifacts = [
        _artifact_entry(artifact) for artifact in sorted(artifacts, key=lambda item: item["id"])
    ]
    built_vulnerability_status = [
        _vulnerability_status_entry(entry, root=root)
        for entry in sorted(vulnerability_status_entries, key=lambda item: item["id"])
    ]
    missing_required = [
        artifact["id"]
        for artifact in built_artifacts
        if artifact["required"] and not artifact["present"]
    ]
    missing_optional = [
        artifact["id"]
        for artifact in built_artifacts
        if not artifact["required"] and not artifact["present"]
    ]
    missing_required_vulnerability_status = [
        entry["id"]
        for entry in built_vulnerability_status
        if entry["required"] and not entry["present"]
    ]
    missing_optional_vulnerability_status = [
        entry["id"]
        for entry in built_vulnerability_status
        if not entry["required"] and not entry["present"]
    ]

    return {
        "schema_version": RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "required_count": sum(int(item["required"]) for item in built_artifacts),
        "optional_count": sum(int(not item["required"]) for item in built_artifacts),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "missing_required_vulnerability_status": missing_required_vulnerability_status,
        "missing_optional_vulnerability_status": missing_optional_vulnerability_status,
        "artifacts": built_artifacts,
        "vulnerability_status": built_vulnerability_status,
        "vulnerability_summary": _vulnerability_summary(built_vulnerability_status),
    }


def _vulnerability_status_entry(raw_entry: dict[str, Any], *, root: Path) -> dict[str, Any]:
    entry_id = str(raw_entry["id"])
    entry_path = str(raw_entry["path"])
    required = bool(raw_entry["required"])
    scanner = str(raw_entry["scanner"])
    resolved_path = _resolve_artifact_root_path(root, entry_path)
    present = resolved_path.exists()
    unresolved = _load_unresolved_vulnerability_summary(resolved_path) if present else {}

    return {
        "id": entry_id,
        "path": str(Path(entry_path)),
        "required": required,
        "scanner": scanner,
        "present": present,
        "unresolved_count": sum(unresolved.values()),
        "unresolved_by_severity": unresolved,
    }


def _load_unresolved_vulnerability_summary(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"UNKNOWN": 1}

    severities: list[str] = []
    if isinstance(payload, list):
        severities.extend(_severity_from_finding(item) for item in payload)
    elif isinstance(payload, dict):
        severities.extend(_trivy_severities(payload))
        severities.extend(_pip_audit_severities(payload))
        severities.extend(_osv_severities(payload))
        if not severities:
            severities.extend(_generic_severities(payload))

    counts: dict[str, int] = {}
    for severity in severities:
        normalized = severity.upper() if severity else "UNKNOWN"
        counts[normalized] = counts.get(normalized, 0) + 1
    return dict(sorted(counts.items()))


def _severity_from_finding(item: Any) -> str:
    if not isinstance(item, dict):
        return "UNKNOWN"
    for key in ("severity", "Severity", "cvss_severity"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return "UNKNOWN"


def _trivy_severities(payload: dict[str, Any]) -> list[str]:
    severities: list[str] = []
    results = payload.get("Results", [])
    if not isinstance(results, list):
        return severities
    for result in results:
        if not isinstance(result, dict):
            continue
        vulnerabilities = result.get("Vulnerabilities", [])
        if not isinstance(vulnerabilities, list):
            continue
        severities.extend(_severity_from_finding(item) for item in vulnerabilities)
    return severities


def _pip_audit_severities(payload: dict[str, Any]) -> list[str]:
    severities: list[str] = []
    dependencies = payload.get("dependencies", [])
    if not isinstance(dependencies, list):
        return severities
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        vulns = dependency.get("vulns", [])
        if isinstance(vulns, list):
            severities.extend(_severity_from_finding(item) for item in vulns)
    return severities


def _osv_severities(payload: dict[str, Any]) -> list[str]:
    severities: list[str] = []
    results = payload.get("results", [])
    if not isinstance(results, list):
        return severities
    for result in results:
        if not isinstance(result, dict):
            continue
        packages = result.get("packages", [])
        if not isinstance(packages, list):
            continue
        for package in packages:
            if not isinstance(package, dict):
                continue
            vulns = package.get("vulnerabilities", [])
            if isinstance(vulns, list):
                severities.extend(_severity_from_finding(item) for item in vulns)
    return severities


def _generic_severities(payload: dict[str, Any]) -> list[str]:
    for key in ("vulnerabilities", "vulns", "findings", "alerts"):
        entries = payload.get(key)
        if isinstance(entries, list):
            return [_severity_from_finding(item) for item in entries]
    return []


def _vulnerability_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    unresolved_by_severity: dict[str, int] = {}
    present_status_count = 0
    for entry in entries:
        if entry["present"]:
            present_status_count += 1
        by_severity = entry.get("unresolved_by_severity", {})
        if not isinstance(by_severity, dict):
            continue
        for severity, count in by_severity.items():
            unresolved_by_severity[str(severity)] = unresolved_by_severity.get(
                str(severity), 0
            ) + int(count)
    return {
        "present_status_count": present_status_count,
        "unresolved_count": sum(unresolved_by_severity.values()),
        "unresolved_by_severity": dict(sorted(unresolved_by_severity.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to release artifact manifest JSON.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory used when resolving artifact paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the generated artifact index JSON.",
    )
    parser.add_argument(
        "--fail-on-missing-required",
        action="store_true",
        help=("Return non-zero if any required artifact is missing from the given root."),
    )
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"failed to load manifest {args.manifest}: {exc}", file=sys.stderr)
        return 1

    manifest_report = validate_artifact_manifest(manifest)
    if not manifest_report["passed"]:
        print(json.dumps(manifest_report, indent=2, sort_keys=True))
        return 1

    artifact_index = build_artifact_index(manifest, root=args.root)
    output_payload = json.dumps(artifact_index, indent=2, sort_keys=True)
    if args.output is None:
        print(output_payload)
    else:
        try:
            args.output.write_text(output_payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"failed to write artifact index to {args.output}: {exc}", file=sys.stderr)
            return 1

    if args.fail_on_missing_required and (
        artifact_index["missing_required"]
        or artifact_index["missing_required_vulnerability_status"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
